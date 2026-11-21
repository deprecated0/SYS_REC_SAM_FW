"""Стадия 9. Сквозной перезапуск пайплайна на подвыборке пользователей."""

import time

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix

from .config import DATA_DIR, SPLIT_DIR, TABLES_DIR, ensure_dirs, load_params
from .io_utils import (
    build_index_maps,
    build_interaction_matrices,
    load_raw_data,
    make_id2idx,
    prepare_book_tags,
)
from .metrics import fast_metrics, rel_info
from .models import (
    META_TAGS,
    build_book_profiles,
    build_favorites,
    build_pool,
    fit_item_profiles,
    fit_svd,
    item_similarity,
    make_content_scorer,
    make_itemcf_scorer,
    make_svd_scorer,
    popularity_scores,
    weight_combos,
)
from .features import (
    FEAT_ORDER,
    build_book_features,
    build_pair_context,
    build_user_features,
    get_primary_author_ids,
    pair_features,
)
from .nn_reranker import nn_scores, train_reranker


def run_pipeline(ratings_path, user_sample, params, verbose=True):
    def log(msg):
        if verbose:
            print(msg)

    df = pd.read_csv(ratings_path).drop_duplicates(["user_id", "book_id"], keep="first")
    df = df[df.user_id.isin(user_sample)].reset_index(drop=True)
    log(f"[1/7] оценок: {len(df):,}, пользователей: {df.user_id.nunique():,}")

    ratings_full, books, tags, book_tags = load_raw_data()
    user_ids_full, book_ids = build_index_maps(ratings_full)
    n_books = len(book_ids)
    bid2i = make_id2idx(book_ids)
    uids = np.sort(df.user_id.unique()).astype(int)
    u2i = dict(zip(uids, np.arange(len(uids))))
    nu = len(uids)

    bt = prepare_book_tags(books, book_tags, tags)
    profiles = build_book_profiles(books, bt, book_ids, bid2i, META_TAGS)
    item_profiles = fit_item_profiles(profiles, min_df=params["content"]["min_df"])

    def split_local(df_in):
        counts = df_in.user_id.value_counts()
        eligible = counts.index[counts >= params["min_train_ratings"]]
        sh = (
            df_in[df_in.user_id.isin(eligible)]
            .sample(frac=1.0, random_state=params["random_state"])
            .reset_index(drop=True)
        )
        sh["rk"] = sh.groupby("user_id").cumcount()
        vl = sh[sh.rk < params["n_val"]].drop(columns="rk")
        ts = sh[
            (sh.rk >= params["n_val"]) & (sh.rk < params["n_val"] + params["n_test"])
        ].drop(columns="rk")
        tr = pd.concat(
            [
                sh[sh.rk >= params["n_val"] + params["n_test"]].drop(columns="rk"),
                df_in[~df_in.user_id.isin(eligible)],
            ],
            ignore_index=True,
        )
        return tr, vl, ts

    tr, vl, ts = split_local(df)
    for d in (tr, vl, ts):
        d["user_idx"] = d.user_id.map(u2i)
        d["book_idx"] = d.book_id.map(bid2i)

    R_tr, _ = build_interaction_matrices(tr, nu, n_books, params["relevance_threshold"])
    R_seen = R_tr.sign().astype(np.bool_)
    uf = build_user_features(tr, nu, params["relevance_threshold"])
    bf, _ = build_book_features(
        tr,
        item_profiles,
        book_ids,
        books,
        n_books,
        params["features"]["n_clusters"],
        params["random_state"],
    )
    author_id = get_primary_author_ids(books, book_ids)
    ctx = build_pair_context(
        R_tr.sign(), bf.cluster.values, n_books, author_id, int(author_id.max()) + 1
    )
    log("[2/7] признаки готовы")

    fav = build_favorites(tr, nu, n_books, params["relevance_threshold"])
    pop_vec, _ = popularity_scores(tr, n_books, params["min_ratings_popular"])
    fns = {
        "content": make_content_scorer(fav, item_profiles, pop_vec),
        "itemcf": make_itemcf_scorer(
            R_tr,
            item_similarity(
                R_tr, params["shrinkage_lambda"], params["top_k_neighbours"]
            ),
            np.full(n_books, tr.rating.mean(), np.float32),
        ),
        "svd_pq": make_svd_scorer(
            fit_svd(
                tr,
                uids,
                book_ids,
                params["svd"]["n_factors"],
                params["pipeline"]["svd_epochs"],
                params["random_state"],
            )
        ),
        "pop": lambda idx: np.broadcast_to(pop_vec, (len(idx), n_books)).copy(),
    }
    log("[3/7] модели обучены")

    def rel_of(d):
        m = (d.rating >= params["relevance_threshold"]).values
        return csr_matrix(
            (
                np.ones(m.sum(), np.float32),
                (d.user_idx.values[m], d.book_idx.values[m]),
            ),
            shape=(nu, n_books),
        )

    rel_v, rel_t = rel_of(vl), rel_of(ts)
    users_eval = np.array(sorted(set(vl.user_idx.values) | set(ts.user_idx.values)))
    pool_per = params["hybrid"]["pool_per_model"]
    pool_b, pool_z = build_pool(
        users_eval, fns, R_seen, n_books, pool_per, len(fns) * pool_per, params["chunk"]
    )
    log(f"[4/7] пул: {(pool_b >= 0).sum(1).mean():.0f} книг на пользователя")

    combos = weight_combos(params["hybrid"]["weight_step"])
    best_w, best_v = None, -1
    val_mask = np.asarray(rel_v.sum(1)).ravel()[users_eval] > 0
    for w in combos:
        m = fast_metrics(
            users_eval[val_mask],
            pool_b[val_mask],
            np.tensordot(pool_z[val_mask], np.asarray(w, np.float32), axes=([2], [0])),
            rel_v,
            params["n_recs"],
        )
        if m["nDCG@10"] > best_v:
            best_v, best_w = m["nDCG@10"], w
    hyb = np.tensordot(pool_z, np.asarray(best_w, np.float32), axes=([2], [0]))
    log(f"[5/7] веса гибрида: {dict(zip(fns, best_w))}")

    seen_c = R_tr.sign().tocoo()
    seen_sets = {}
    for u, b in zip(seen_c.row, seen_c.col):
        seen_sets.setdefault(u, []).append(b)
    hi_lists = (
        tr[tr.rating >= params["relevance_threshold"]]
        .groupby("user_idx")["book_idx"]
        .apply(list)
    )
    rng_l = np.random.default_rng(params["random_state"])
    pu_, pb_, py_ = [], [], []
    for u in range(nu):
        fav = hi_lists.get(u, [])
        if not fav:
            continue
        pos = rng_l.choice(
            fav, size=min(params["reranker"]["n_pos"], len(fav)), replace=False
        )
        cand = rng_l.choice(n_books, size=49, replace=False)
        neg = cand[~np.isin(cand, seen_sets.get(u, []))][: params["reranker"]["n_neg"]]
        pu_ += [u] * (len(pos) + len(neg))
        pb_ += list(pos) + list(neg)
        py_ += [1] * len(pos) + [0] * len(neg)
    pu_ = np.array(pu_, np.int64)
    pb_ = np.array(pb_, np.int64)
    py_ = np.array(py_, np.float32)
    X_l = np.stack(
        [
            pair_features(
                pu_,
                pb_,
                ctx,
                author_id,
                uf,
                bf,
                model_score_fns={
                    "content": fns["content"],
                    "itemcf": fns["itemcf"],
                    "svd_pq": fns["svd_pq"],
                },
            )[k]
            for k in FEAT_ORDER
        ],
        axis=1,
    ).astype(np.float32)
    torch.manual_seed(params["random_state"])
    reranker = train_reranker(
        X_l, pu_, pb_, py_, nu, n_books, params["pipeline"]["nn_epochs"], use_deep=True
    )
    log("[6/7] реранкер обучен")

    t_mask = np.asarray(rel_t.sum(1)).ravel()[users_eval] > 0
    ue_t = users_eval[t_mask]
    rows = {}
    for name, w in zip(
        ["content", "itemcf", "svd_pq", "pop"],
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    ):
        rows[name] = fast_metrics(
            ue_t,
            pool_b[t_mask],
            np.tensordot(pool_z[t_mask], np.asarray(w, np.float32), axes=([2], [0])),
            rel_t,
            params["n_recs"],
        )["nDCG@10"]
    rows["hybrid"] = fast_metrics(
        ue_t, pool_b[t_mask], hyb[t_mask], rel_t, params["n_recs"]
    )["nDCG@10"]
    rerank_pool = params["reranker"]["rerank_pool"]
    n_recs = params["n_recs"]
    base = np.where(pool_b[t_mask] < 0, -np.inf, hyb[t_mask])
    part = np.argpartition(-base, rerank_pool - 1, axis=1)[:, :rerank_pool]
    cand = np.take_along_axis(pool_b[t_mask], part, axis=1)
    u_flat = np.repeat(ue_t, rerank_pool)
    b_flat = cand.ravel()
    z_flat = np.take_along_axis(pool_z[t_mask], part[:, :, None], axis=1)
    feats = np.stack(
        [z_flat[:, :, i].ravel() for i in range(3)]
        + [
            pair_features(u_flat, b_flat, ctx, author_id, uf, bf)[k]
            for k in FEAT_ORDER[3:]
        ],
        axis=1,
    ).astype(np.float32)
    sc = nn_scores(reranker, feats, u_flat, b_flat).reshape(len(ue_t), rerank_pool)
    order = np.argsort(-sc, axis=1)[:, :n_recs]
    top_books = np.take_along_axis(cand, order, axis=1)
    hits = (
        np.asarray(rel_t[np.repeat(ue_t, n_recs), top_books.ravel()])
        .ravel()
        .astype(np.float64)
        .reshape(len(ue_t), n_recs)
    )
    n_rel = np.asarray(rel_t.sum(1)).ravel()[ue_t]
    disc = 1.0 / np.log2(np.arange(2, n_recs + 2))
    disc_cum = np.concatenate([[0], np.cumsum(disc)])
    rows["hybrid + W&D rerank"] = float(
        ((hits * disc).sum(1) / disc_cum[np.clip(n_rel, 0, n_recs).astype(int)]).mean()
    )
    log("[7/7] готово")
    return pd.Series(rows, name="nDCG@10 (перезапуск пайплайна)")


def main():
    params = load_params()
    ensure_dirs()
    ratings, _, _, _ = load_raw_data()
    rng = np.random.default_rng(params["random_state"])
    sample = np.sort(
        rng.choice(
            ratings.user_id.unique(),
            size=params["pipeline"]["user_sample"],
            replace=False,
        )
    )
    t0 = time.perf_counter()
    result = run_pipeline(DATA_DIR / "ratings.csv", sample, params)
    print(f"Время: {time.perf_counter() - t0:.0f} с")
    result.round(4).to_csv(TABLES_DIR / "pipeline_result.csv", index_label="model")
    print(result.round(4))


if __name__ == "__main__":
    main()
