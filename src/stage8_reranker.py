"""Стадия 8. Нейросетевой реранкер Wide & Deep поверх гибридного пула."""

import time

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix, load_npz
from sklearn.metrics import roc_auc_score

from .config import (
    DATA_DIR,
    MODELS_DIR,
    SPLIT_DIR,
    TABLES_DIR,
    ensure_dirs,
    load_params,
)
from .io_utils import load_index_maps, load_raw_data, build_interaction_matrices
from .features import (
    FEAT_ORDER,
    build_book_features,
    build_pair_context,
    build_user_features,
    get_primary_author_ids,
    pair_features,
)
from .metrics import fast_metrics, rel_info
from .models import make_content_scorer, make_itemcf_scorer, make_svd_scorer
from .nn_reranker import WideDeep, make_training_pairs, nn_scores, train_reranker


def main():
    params = load_params()
    ensure_dirs()
    rp = params["reranker"]
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
    pool_users = np.load(MODELS_DIR / "pool_users.npy")
    pool_books = np.load(MODELS_DIR / "pool_books.npy")
    pool_z = np.load(MODELS_DIR / "pool_z.npy")
    _, books, _, _ = load_raw_data()
    i_pool = {int(u): i for i, u in enumerate(pool_users)}

    def pop_score(idx):
        return np.broadcast_to(pop_scores, (len(idx), n_books)).copy()

    content_score = make_content_scorer(fav_train, item_profiles, pop_scores)
    itemcf_score = make_itemcf_scorer(R_train, S_topk, item_mean)
    svd_score = make_svd_scorer(svd_bundle)

    def svd_pq_score(idx):
        return svd_bundle["pu"][idx] @ svd_bundle["qi"].T

    # признаки
    uf = build_user_features(train, n_users, params["relevance_threshold"])
    bf, _ = build_book_features(
        train,
        item_profiles,
        book_ids,
        books,
        n_books,
        params["features"]["n_clusters"],
        params["random_state"],
    )
    author_id = get_primary_author_ids(books, book_ids)
    ctx = build_pair_context(
        R_train.sign(), bf.cluster.values, n_books, author_id, int(author_id.max()) + 1
    )

    # обучающие пары
    torch.manual_seed(params["random_state"])
    t0 = time.perf_counter()
    pair_u, pair_b, pair_y = make_training_pairs(
        train,
        R_train,
        n_users,
        n_books,
        rp["n_pos"],
        rp["n_neg"],
        params["relevance_threshold"],
        params["random_state"],
    )
    print(
        f"Пар для обучения: {len(pair_y):,} (позитивов {pair_y.sum():.0f}) за {time.perf_counter() - t0:.0f} с"
    )
    t0 = time.perf_counter()
    nn_feats = pair_features(
        pair_u,
        pair_b,
        ctx,
        author_id,
        uf,
        bf,
        model_score_fns={
            "content": content_score,
            "itemcf": itemcf_score,
            "svd_pq": svd_pq_score,
        },
    )
    print(f"Признаки за {time.perf_counter() - t0:.0f} с")
    X = np.stack([nn_feats[k] for k in FEAT_ORDER], axis=1).astype(np.float32)

    # обучение двух вариантов
    print("Полный Wide & Deep:")
    wd_model = train_reranker(
        X,
        pair_u,
        pair_b,
        pair_y,
        n_users,
        n_books,
        rp["n_epochs"],
        rp["batch"],
        rp["lr"],
        use_deep=True,
    )
    print("Абляция wide-only:")
    wide_only = train_reranker(
        X,
        pair_u,
        pair_b,
        pair_y,
        n_users,
        n_books,
        rp["n_epochs"],
        rp["batch"],
        rp["lr"],
        use_deep=False,
    )
    torch.save(wd_model.state_dict(), MODELS_DIR / "reranker_wide_deep.pt")
    torch.save(wide_only.state_dict(), MODELS_DIR / "reranker_wide_only.pt")

    # AUC на валидации
    hi_val = val[val.rating >= params["relevance_threshold"]]
    lo_val = val[val.rating < params["relevance_threshold"]]
    rng_v = np.random.default_rng(params["random_state"])
    s_hi = hi_val.sample(min(3000, len(hi_val)), random_state=params["random_state"])
    s_lo = lo_val.sample(min(3000, len(lo_val)), random_state=params["random_state"])
    val_u = np.concatenate([s_hi.user_idx.values, s_lo.user_idx.values])
    val_b = np.concatenate([s_hi.book_idx.values, s_lo.book_idx.values])
    val_y = np.concatenate([np.ones(len(s_hi)), np.zeros(len(s_lo))]).astype(np.float32)
    Xval = np.stack(
        [
            pair_features(
                val_u,
                val_b,
                ctx,
                author_id,
                uf,
                bf,
                model_score_fns={
                    "content": content_score,
                    "itemcf": itemcf_score,
                    "svd_pq": svd_pq_score,
                },
            )[k]
            for k in FEAT_ORDER
        ],
        axis=1,
    ).astype(np.float32)
    auc_full = roc_auc_score(val_y, nn_scores(wd_model, Xval, val_u, val_b))
    auc_wide = roc_auc_score(val_y, nn_scores(wide_only, Xval, val_u, val_b))
    pd.DataFrame(
        {"model": ["Wide & Deep", "wide-only"], "AUC_val": [auc_full, auc_wide]}
    ).to_csv(TABLES_DIR / "auc_table.csv", index=False)
    print(f"AUC на валидации: полный={auc_full:.4f}, wide-only={auc_wide:.4f}")

    # реранкинг на тесте
    rel_sets_t, eval_users_t, rel_test = rel_info(
        test, n_users, n_books, params["relevance_threshold"]
    )
    best_w = (
        pd.read_csv(TABLES_DIR / "hybrid_weights.csv")
        .iloc[0][["w_content", "w_itemcf", "w_svd_pq", "w_pop"]]
        .values.astype(float)
    )
    sub_test = np.array([i_pool[int(u)] for u in eval_users_t])
    hyb = np.tensordot(pool_z[sub_test], best_w, axes=([2], [0]))
    n_recs = params["n_recs"]
    rerank_pool = rp["rerank_pool"]

    def rerank_metrics(model, name):
        base = np.where(pool_books[sub_test] < 0, -np.inf, hyb)
        part = np.argpartition(-base, rerank_pool - 1, axis=1)[:, :rerank_pool]
        cand = np.take_along_axis(pool_books[sub_test], part, axis=1)
        u_flat = np.repeat(eval_users_t, rerank_pool)
        b_flat = cand.ravel()
        z_flat = np.take_along_axis(pool_z[sub_test], part[:, :, None], axis=1)
        other = pair_features(u_flat, b_flat, ctx, author_id, uf, bf)
        feats = np.stack(
            [z_flat[:, :, i].ravel() for i in range(3)]
            + [other[k] for k in FEAT_ORDER[3:]],
            axis=1,
        ).astype(np.float32)
        sc = nn_scores(model, feats, u_flat, b_flat).reshape(
            len(eval_users_t), rerank_pool
        )
        order = np.argsort(-sc, axis=1)[:, :n_recs]
        top_part = np.take_along_axis(part, order, axis=1)
        top_books = np.take_along_axis(pool_books[sub_test], top_part, axis=1)
        hits = (
            np.asarray(rel_test[np.repeat(eval_users_t, n_recs), top_books.ravel()])
            .ravel()
            .astype(np.float64)
            .reshape(len(eval_users_t), n_recs)
        )
        n_rel = np.asarray(rel_test.sum(1)).ravel()[eval_users_t]
        disc = 1.0 / np.log2(np.arange(2, n_recs + 2))
        return {
            "model": name,
            "P@10": float((hits.sum(1) / n_recs).mean()),
            "R@10": float(
                np.divide(
                    hits.sum(1), n_rel, out=np.zeros_like(n_rel), where=n_rel > 0
                ).mean()
            ),
            "nDCG@10": float(
                (
                    (hits * disc).sum(1)
                    / np.concatenate([[0], np.cumsum(disc)])[
                        np.clip(n_rel, 0, n_recs).astype(int)
                    ]
                ).mean()
            ),
        }

    wd_row = rerank_metrics(wd_model, "Гибрид + Wide&Deep реранк")
    wide_row = rerank_metrics(wide_only, "Гибрид + wide-only реранк")
    metrics_hybrid = pd.read_csv(TABLES_DIR / "metrics_hybrid.csv", index_col="model")
    metrics_final = pd.concat(
        [
            metrics_hybrid,
            pd.DataFrame([wd_row, wide_row]).set_index("model"),
        ]
    )
    metrics_final.to_csv(TABLES_DIR / "metrics_reranker.csv", index_label="model")
    print(metrics_final[["P@10", "R@10", "nDCG@10"]])
    print(f"Лучшая система по nDCG@10: {metrics_final['nDCG@10'].idxmax()}")


if __name__ == "__main__":
    main()
