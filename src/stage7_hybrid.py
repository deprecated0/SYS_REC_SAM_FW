"""Стадия 7. Гибридный пул кандидатов, подбор весов на валидации, сегменты, диверсификация."""

import itertools
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz

from .config import (
    DATA_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    SPLIT_DIR,
    TABLES_DIR,
    ensure_dirs,
    load_params,
)
from .io_utils import (
    build_interaction_matrices,
    load_index_maps,
    load_raw_data,
    save_fig,
    setup_plots,
)
from .metrics import fast_metrics, rel_info
from .models import (
    build_pool,
    make_content_scorer,
    make_itemcf_scorer,
    make_svd_scorer,
    weight_combos,
    z_norm_row,
)
from .features import (
    build_book_features,
    build_user_features,
    get_primary_author_ids,
    build_pair_context,
)


def diversify(
    users_arr, scores, pool_books, author_id, book_cluster, max_a, max_c, k, i_pool
):
    ixs = np.array([i_pool[u] for u in users_arr])
    order = np.argsort(-np.where(pool_books[ixs] < 0, -np.inf, scores), axis=1)
    result = np.full((len(ixs), k), -1, np.int32)
    for r, ix in enumerate(ixs):
        chosen, a_cnt, c_cnt = [], {}, {}
        for p in order[r]:
            b = pool_books[ix, p]
            if b < 0:
                break
            a, c = author_id[b], book_cluster[b]
            if a_cnt.get(a, 0) >= max_a or c_cnt.get(c, 0) >= max_c:
                continue
            chosen.append(p)
            a_cnt[a] = a_cnt.get(a, 0) + 1
            c_cnt[c] = c_cnt.get(c, 0) + 1
            if len(chosen) == k:
                break
        result[r, : len(chosen)] = chosen
    return result


def main():
    params = load_params()
    ensure_dirs()
    setup_plots()
    train = pd.read_csv(SPLIT_DIR / "train.csv")
    val = pd.read_csv(SPLIT_DIR / "val.csv")
    test = pd.read_csv(SPLIT_DIR / "test.csv")
    user_ids, book_ids = load_index_maps(SPLIT_DIR / "index_maps.npz")
    n_users, n_books = len(user_ids), len(book_ids)
    R_train = load_npz(MODELS_DIR / "R_train.npz")
    S_topk = load_npz(MODELS_DIR / "S_topk.npz")
    item_mean = np.load(MODELS_DIR / "item_mean.npy")
    pop_scores = np.load(MODELS_DIR / "pop_scores.npy")
    item_profiles = load_npz(MODELS_DIR / "item_profiles.npz")
    fav_train = load_npz(MODELS_DIR / "fav_train.npz")
    svd_bundle = np.load(MODELS_DIR / "svd_bundle.npz")
    _, books, tags, book_tags = load_raw_data()

    def pop_score(idx):
        return np.broadcast_to(pop_scores, (len(idx), n_books)).copy()

    content_score = make_content_scorer(fav_train, item_profiles, pop_scores)
    itemcf_score = make_itemcf_scorer(R_train, S_topk, item_mean)
    svd_score = make_svd_scorer(svd_bundle)

    def svd_pq_score(idx):
        return svd_bundle["pu"][idx] @ svd_bundle["qi"].T

    pool_fns = {
        "content": content_score,
        "itemcf": itemcf_score,
        "svd_pq": svd_pq_score,
        "pop": pop_score,
    }
    pool_per_model = params["hybrid"]["pool_per_model"]
    pool_pad = len(pool_fns) * pool_per_model
    R_seen = R_train.sign().astype(np.bool_)

    pool_users = np.array(sorted(set(val.user_idx.values) | set(test.user_idx.values)))
    print(f"Пул для {len(pool_users):,} пользователей, {pool_per_model} на модель")
    t0 = time.perf_counter()
    pool_books, pool_z = build_pool(
        pool_users, pool_fns, R_seen, n_books, pool_per_model, pool_pad, params["chunk"]
    )
    print(
        f"Пул построен за {time.perf_counter() - t0:.0f} с, средний размер: {(pool_books >= 0).sum(1).mean():.0f}"
    )

    np.save(MODELS_DIR / "pool_users.npy", pool_users)
    np.save(MODELS_DIR / "pool_books.npy", pool_books)
    np.save(MODELS_DIR / "pool_z.npy", pool_z)

    rel_mask_val = (val.rating >= params["relevance_threshold"]).values
    rel_val = csr_matrix(
        (
            np.ones(rel_mask_val.sum(), np.float32),
            (val.user_idx.values[rel_mask_val], val.book_idx.values[rel_mask_val]),
        ),
        shape=(n_users, n_books),
    )
    rel_mask_test = (test.rating >= params["relevance_threshold"]).values
    rel_test = csr_matrix(
        (
            np.ones(rel_mask_test.sum(), np.float32),
            (test.user_idx.values[rel_mask_test], test.book_idx.values[rel_mask_test]),
        ),
        shape=(n_users, n_books),
    )
    rel_sets_val = {
        u: set(rel_val.getrow(u).indices)
        for u in np.unique(val.user_idx.values[rel_mask_val])
    }
    rel_sets_test = {
        u: set(rel_test.getrow(u).indices)
        for u in np.unique(test.user_idx.values[rel_mask_test])
    }
    eval_users_v = np.array(sorted(rel_sets_val))
    eval_users_t = np.array(sorted(rel_sets_test))

    i_pool = {int(u): i for i, u in enumerate(pool_users)}

    # подбор весов на валидации
    combos = weight_combos(params["hybrid"]["weight_step"])
    print(f"Комбинаций весов: {len(combos)}")
    val_users_rel = np.array(
        sorted(set(val.user_idx.values[val.rating >= params["relevance_threshold"]]))
    )
    sub_val = np.array([i_pool[int(u)] for u in val_users_rel])
    grid = []
    for w in combos:
        sc = np.tensordot(pool_z[sub_val], np.asarray(w, np.float32), axes=([2], [0]))
        m = fast_metrics(
            val_users_rel, pool_books[sub_val], sc, rel_val, params["n_recs"]
        )
        grid.append((*w, m["nDCG@10"]))
    grid_df = pd.DataFrame(
        grid, columns=["w_content", "w_itemcf", "w_svd_pq", "w_pop", "nDCG@10_val"]
    )
    grid_df = grid_df.sort_values("nDCG@10_val", ascending=False).reset_index(drop=True)
    grid_df.to_csv(TABLES_DIR / "hybrid_weights.csv", index=False)
    best_w = grid_df.iloc[0][
        ["w_content", "w_itemcf", "w_svd_pq", "w_pop"]
    ].values.astype(float)
    print(
        f"Лучшие веса: {dict(zip(pool_fns, best_w))}, nDCG@10={grid_df.iloc[0]['nDCG@10_val']:.4f}"
    )

    # сегментные веса
    uf = build_user_features(train, n_users, params["relevance_threshold"])
    seg_of_pool = uf.segment.values[pool_users]
    seg_val = seg_of_pool[sub_val]
    seg_rows = []
    seg_weights = {}
    for seg in ["cold", "medium", "active"]:
        mask = seg_val == seg
        if mask.sum() < 300:
            seg_weights[seg] = best_w
            continue
        best_local, best_v = None, -1
        for w in combos:
            sc = np.tensordot(
                pool_z[sub_val[mask]], np.asarray(w, np.float32), axes=([2], [0])
            )
            m = fast_metrics(
                val_users_rel[mask],
                pool_books[sub_val[mask]],
                sc,
                rel_val,
                params["n_recs"],
            )
            if m["nDCG@10"] > best_v:
                best_v, best_local = m["nDCG@10"], w
        seg_weights[seg] = np.array(best_local)
        seg_rows.append(
            {
                "segment": seg,
                "users": int(mask.sum()),
                **{f"w_{n}": w for n, w in zip(pool_fns, best_local)},
                "nDCG@10_val": best_v,
            }
        )
    pd.DataFrame(seg_rows).to_csv(TABLES_DIR / "seg_weights.csv", index=False)

    # оценка на тесте
    sub_test = np.array([i_pool[int(u)] for u in eval_users_t])
    hyb_simple = np.tensordot(pool_z[sub_test], best_w, axes=([2], [0]))
    seg_arr = uf.segment.values[eval_users_t]
    hyb_seg = np.zeros((len(eval_users_t), pool_pad), np.float32)
    seg_arr_clean = np.array(
        [str(s) if pd.notna(s) else "unknown" for s in seg_arr], dtype=object
    )
    for seg in np.unique(seg_arr_clean):
        m = seg_arr_clean == seg
        w = seg_weights.get(seg, best_w)
        hyb_seg[m] = np.tensordot(
            pool_z[sub_test[m]], np.asarray(w, np.float32), axes=([2], [0])
        )

    results_main = pd.read_csv(TABLES_DIR / "metrics_main.csv", index_col="model")
    n_recs = params["n_recs"]
    for name, sc in [
        ("Гибрид (общие веса)", hyb_simple),
        ("Гибрид (веса по сегментам)", hyb_seg),
    ]:
        m = fast_metrics(eval_users_t, pool_books[sub_test], sc, rel_test, n_recs)
        results_main.loc[name] = {
            "P@5": m["P@5"],
            "R@5": m["R@5"],
            "nDCG@5": m["nDCG@5"],
            "P@10": m["P@10"],
            "R@10": m["R@10"],
            "nDCG@10": m["nDCG@10"],
        }
    results_main.to_csv(TABLES_DIR / "metrics_hybrid.csv", index_label="model")
    print(results_main)

    ax = (
        results_main[["P@10", "R@10", "nDCG@10"]]
        .sort_values("nDCG@10")
        .plot(kind="barh", figsize=(10, 5.5), colormap="crest")
    )
    ax.set_title("Итоговое сравнение моделей и гибридов (тест)")
    ax.set_xlabel("Значение метрики")
    ax.set_ylabel("Модель")
    ax.figure.tight_layout()
    save_fig(ax.figure, "metrics_final.png")

    # сегментный разбор
    bf, km = build_book_features(
        train,
        item_profiles,
        book_ids,
        books,
        n_books,
        params["features"]["n_clusters"],
        params["random_state"],
    )
    seg_test = np.array(
        [str(s) if pd.notna(s) else "unknown" for s in uf.segment.values[eval_users_t]],
        dtype=object,
    )
    seg_table = {}
    for label, w in [
        ("Popularity", [0, 0, 0, 1]),
        ("Content (TF-IDF)", [1, 0, 0, 0]),
        ("ItemKNN CF", [0, 1, 0, 0]),
        ("Гибрид (сегменты)", None),
    ]:
        row = {}
        for seg in ["cold", "medium", "active"]:
            m = seg_test == seg
            if w is not None:
                sc = np.tensordot(
                    pool_z[sub_test[m]], np.asarray(w, np.float32), axes=([2], [0])
                )
            else:
                sc = hyb_seg[m]
            row[seg] = fast_metrics(
                eval_users_t[m], pool_books[sub_test[m]], sc, rel_test, n_recs
            )["nDCG@10"]
        seg_table[label] = row
    seg_df = pd.DataFrame(seg_table).T
    seg_df.to_csv(TABLES_DIR / "segment_table.csv")
    ax2 = seg_df.plot(kind="bar", figsize=(10, 4.5), colormap="crest", rot=10)
    ax2.set_title("nDCG@10 по сегментам пользователей (тест)")
    ax2.set_xlabel("Модель")
    ax2.set_ylabel("nDCG@10")
    ax2.figure.tight_layout()
    save_fig(ax2.figure, "segment_bar.png")

    # диверсификация
    author_id = get_primary_author_ids(books, book_ids)
    div = params["diversity"]
    disc = 1.0 / np.log2(np.arange(2, n_recs + 2))
    disc_cum = np.concatenate([[0.0], np.cumsum(disc)])

    def div_row(name, pos):
        hits = (
            np.asarray(
                rel_test[
                    np.repeat(eval_users_t, n_recs),
                    np.take_along_axis(pool_books[sub_test], pos, axis=1).ravel(),
                ]
            )
            .ravel()
            .astype(np.float64)
            .reshape(len(eval_users_t), n_recs)
        )
        n_rel = np.asarray(rel_test.sum(1)).ravel()[eval_users_t]
        nov_flat = np.take_along_axis(pool_books[sub_test], pos, axis=1).ravel()
        nov_flat = nov_flat[nov_flat >= 0]
        novelty = float(np.log1p(bf.n_ratings.values[nov_flat]).mean())
        coverage = len(np.unique(nov_flat)) / n_books
        auth_shares = []
        for r in range(len(eval_users_t)):
            bks = pool_books[sub_test[r], pos[r][pos[r] >= 0]]
            if len(bks):
                auth_shares.append(np.bincount(author_id[bks]).max() / len(bks))
        return {
            "variant": name,
            "P@10": float((hits.sum(1) / n_recs).mean()),
            "nDCG@10": float(
                (
                    (hits * disc).sum(1)
                    / disc_cum[np.clip(n_rel, 0, n_recs).astype(int)]
                ).mean()
            ),
            "novelty": novelty,
            "coverage": coverage,
            "top_author_share": float(np.mean(auth_shares)),
        }

    base_pos = np.argsort(
        -np.where(pool_books[sub_test] < 0, -np.inf, hyb_seg), axis=1
    )[:, :n_recs]
    div_rows = [div_row("без ограничений", base_pos)]
    for label, ma, mc in [
        ("мягкие лимиты", div["max_per_author"], div["max_per_cluster"]),
        ("жёсткие лимиты", 2, 4),
    ]:
        pos = diversify(
            eval_users_t,
            hyb_seg,
            pool_books,
            author_id,
            bf.cluster.values,
            ma,
            mc,
            n_recs,
            i_pool,
        )
        div_rows.append(div_row(label, pos))
    div_df = pd.DataFrame(div_rows).set_index("variant")
    div_df.to_csv(TABLES_DIR / "diversity_table.csv")
    print(div_df.round(4))


if __name__ == "__main__":
    main()
