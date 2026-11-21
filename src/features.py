"""Признаки пользователей, книг и пар для реранкера."""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans


def build_user_features(train_df, n_users, relevance_threshold):
    g = train_df.groupby("user_idx")["rating"]
    hi = train_df.assign(hi=(train_df.rating >= relevance_threshold).astype(float))
    uf = pd.DataFrame(index=np.arange(n_users))
    uf["n_ratings"] = g.size().reindex(uf.index, fill_value=0)
    uf["mean_rating"] = g.mean().reindex(uf.index, fill_value=train_df.rating.mean())
    uf["std_rating"] = g.std().reindex(uf.index).fillna(0.0)
    uf["share_hi"] = (
        hi.groupby("user_idx")["hi"].mean().reindex(uf.index, fill_value=0.0)
    )
    uf["segment"] = pd.cut(
        uf.n_ratings, [-1, 9, 39, float("inf")], labels=["cold", "medium", "active"]
    )
    return uf


def build_book_features(
    train_df, item_profiles, book_ids, books, n_books, n_clusters, seed
):
    g = train_df.groupby("book_idx")["rating"]
    bf = pd.DataFrame(index=np.arange(n_books))
    bf["n_ratings"] = g.size().reindex(bf.index, fill_value=0).astype(int)
    bf["mean_rating"] = g.mean().reindex(bf.index, fill_value=train_df.rating.mean())
    bf["std_rating"] = g.std().reindex(bf.index).fillna(0.0)
    bf["log_pop"] = np.log1p(bf.n_ratings)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=4)
    bf["cluster"] = km.fit_predict(item_profiles)
    year = books.set_index("id").loc[book_ids, "original_publication_year"].values
    bf["year"] = pd.Series(year).fillna(pd.Series(year).median()).values
    return bf, km


def build_pair_context(R_bin, book_cluster, n_books, author_id, n_authors):
    A = csr_matrix(
        (np.ones(n_books), (np.arange(n_books), author_id)), shape=(n_books, n_authors)
    )
    author_cnt = R_bin @ A
    n_cl = int(book_cluster.max()) + 1
    C = csr_matrix(
        (np.ones(n_books), (np.arange(n_books), book_cluster)), shape=(n_books, n_cl)
    )
    cl_cnt = np.asarray((R_bin @ C).todense())
    cl_pref = cl_cnt / np.clip(cl_cnt.sum(1, keepdims=True), 1, None)
    return {"author_cnt": author_cnt.tocsr(), "cl_pref": cl_pref.astype(np.float32)}


def get_primary_author_ids(books, book_ids):
    primary_author = (
        books.set_index("id").loc[book_ids, "authors"].str.split(",").str[0].str.strip()
    )
    author_id, _ = pd.factorize(primary_author)
    return author_id.astype(np.int64)


FEAT_ORDER = [
    "z_content",
    "z_itemcf",
    "z_svd_pq",
    "same_author",
    "cluster_pref",
    "book_log_pop",
    "book_mean",
    "book_std",
    "book_year",
    "user_log_n",
    "user_mean",
    "user_share_hi",
]


def pair_features(user_idx, book_idx, ctx, author_id, uf, bf, model_score_fns=None):
    f = {}
    f["same_author"] = np.asarray(
        ctx["author_cnt"][user_idx, author_id[book_idx]]
    ).ravel()
    f["same_author"] = (f["same_author"] > 0).astype(np.float32)
    f["cluster_pref"] = ctx["cl_pref"][user_idx, bf.cluster.values[book_idx]]
    for name, col in [
        ("book_log_pop", "log_pop"),
        ("book_mean", "mean_rating"),
        ("book_std", "std_rating"),
        ("book_year", "year"),
    ]:
        f[name] = bf[col].values[book_idx].astype(np.float32)
    f["book_year"] = (f["book_year"] - 1900) / 100
    f["user_log_n"] = np.log1p(uf.n_ratings.values[user_idx])
    f["user_mean"] = uf.mean_rating.values[user_idx]
    f["user_share_hi"] = uf.share_hi.values[user_idx]
    if model_score_fns:
        for name, fn in model_score_fns.items():
            f["z_" + name] = gather_z_scores(fn, user_idx, book_idx)
    return f


def gather_z_scores(score_fn, user_idx, book_idx, chunk=4096):
    order = np.argsort(user_idx, kind="stable")
    us = user_idx[order]
    out = np.empty(len(user_idx), np.float32)
    uniq = np.unique(us)
    for st in range(0, len(uniq), chunk):
        part = uniq[st : st + chunk]
        s = np.asarray(score_fn(part), np.float32)
        mu = s.mean(1, keepdims=True)
        sd = s.std(1, keepdims=True)
        z = (s - mu) / (sd + 1e-9)
        lo = np.searchsorted(us, part[0])
        hi = np.searchsorted(us, part[-1], "right")
        sel = order[lo:hi]
        out[sel] = z[np.searchsorted(part, user_idx[sel]), book_idx[sel]]
    return out
