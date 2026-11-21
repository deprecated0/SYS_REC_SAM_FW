"""Модели: популярность, TF-IDF, ItemCF, SVD, гибриды."""

import itertools

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from surprise import SVD as SurpriseSVD
from surprise import Dataset, Reader

from .metrics import topn_rows

META_TAGS = {
    "to-read",
    "currently-reading",
    "owned",
    "books-i-own",
    "favorites",
    "favourites",
    "favorite",
    "favourite",
    "favorite-books",
    "favourite-books",
    "default",
    "to-buy",
    "owned-books",
    "my-books",
    "my-library",
    "library",
    "kindle",
    "ebooks",
    "ebook",
    "audiobook",
    "audiobooks",
    "audio",
    "want-to-read",
    "did-not-finish",
    "dnf",
    "book-club",
    "re-read",
    "reread",
    "owned-to-read",
    "i-owned",
    "school",
    "class",
    "favourites-read",
    "favorites-read",
    "books",
    "read",
    "read-in-2015",
    "read-in-2016",
    "read-2015",
    "read-2016",
    "shelfari-favorites",
    "all-time-favorites",
}


def popularity_scores(train_df, n_books, min_ratings):
    stats = train_df.groupby("book_idx").agg(
        mean=("rating", "mean"), n=("rating", "size")
    )
    scores = np.full(n_books, -1.0, dtype=np.float32)
    eligible = stats.index[stats.n >= min_ratings]
    scores[eligible] = stats.loc[eligible, "mean"].values
    return scores, stats


def find_demo_book(books, pattern):
    row = books[books.title.str.contains(pattern, case=False, na=False, regex=False)]
    return int(row["id"].iloc[0])


def build_book_profiles(books, bt, book_ids, bid2i, meta_tags):
    n_books = len(book_ids)
    books_by_id = books.set_index("id")
    titles = books_by_id.loc[book_ids, "original_title"].fillna(
        books_by_id.loc[book_ids, "title"]
    )
    bt = bt.copy()
    bt["book_idx"] = bt["id"].map(bid2i)
    bt = bt[bt["book_idx"].notna()]
    bt_content = bt[~bt["tag_clean"].isin(meta_tags)]
    profile_tags = (
        bt_content.sort_values(["book_idx", "count"], ascending=[True, False])
        .groupby("book_idx")["tag_clean"]
        .agg(" ".join)
    )
    titles_clean = (
        titles.str.lower().str.replace(r"[^\w\s]", " ", regex=True).fillna("")
    )
    tags_text = profile_tags.reindex(np.arange(n_books)).fillna("")
    return titles_clean.reset_index(drop=True).str.cat(
        tags_text.reset_index(drop=True), sep=" "
    )


def fit_item_profiles(profiles, min_df=3):
    tfidf = TfidfVectorizer(
        min_df=min_df, sublinear_tf=True, token_pattern=r"(?u)\b\w\w+\b"
    )
    return normalize(tfidf.fit_transform(profiles.values))


def get_similar_books(item_profiles, idx, n):
    sims = (item_profiles @ item_profiles[idx].T).toarray().ravel()
    sims[idx] = -1.0
    top = np.argsort(-sims)[:n]
    return top, sims[top]


def build_favorites(df, n_users, n_books, relevance_threshold):
    fav = df[df.rating >= relevance_threshold]
    no_fav = set(df.user_idx.unique()) - set(fav.user_idx.unique())
    if no_fav:
        fav = pd.concat([fav, df[df.user_idx.isin(no_fav)]])
    w = 1.0 / fav.groupby("user_idx")["book_idx"].transform("size")
    return csr_matrix(
        (w.values, (fav.user_idx.values, fav.book_idx.values)),
        shape=(n_users, n_books),
    )


def make_content_scorer(fav_csr, item_profiles, fallback_scores):
    def score_fn(user_indices):
        prof = fav_csr[user_indices] @ item_profiles
        scores = np.asarray((prof @ item_profiles.T).todense(), dtype=np.float32)
        empty = np.asarray(prof.getnnz(axis=1) == 0)
        if empty.any():
            scores[empty] = fallback_scores
        return scores

    return score_fn


def keep_topk_per_row(S, k):
    S = S.tocsr()
    rows, cols, vals = [], [], []
    indptr, indices, data = S.indptr, S.indices, S.data
    for i in range(S.shape[0]):
        lo, hi = indptr[i], indptr[i + 1]
        sel = (
            np.argpartition(-data[lo:hi], k)[:k] if hi - lo > k else np.arange(hi - lo)
        )
        rows.append(np.full(len(sel), i))
        cols.append(indices[lo:hi][sel])
        vals.append(data[lo:hi][sel])
    return csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=S.shape,
    )


def item_similarity(R, lmbda=10.0, k=20):
    item_users = normalize(R.T.tocsr())
    S = (item_users @ item_users.T).tocsr()
    co = (R.sign().T @ R.sign()).tocsr()
    shrink = co.copy()
    shrink.data = shrink.data / (shrink.data + lmbda)
    S = S.multiply(shrink).tocsr()
    S.setdiag(0.0)
    S.eliminate_zeros()
    return keep_topk_per_row(S, k)


def predict_rating_itemcf(user_idx, book_idx, S_topk, R_train, item_mean, k):
    neighbours = S_topk.getrow(book_idx)
    order = np.argsort(-neighbours.data)[:k]
    nb_idx, nb_sim = neighbours.indices[order], neighbours.data[order]
    user_row = R_train.getrow(user_idx)
    rated = dict(zip(user_row.indices, user_row.data))
    common = [(j, s) for j, s in zip(nb_idx, nb_sim) if j in rated]
    if not common:
        return float(item_mean[book_idx]), 0
    num = sum(s * rated[j] for j, s in common)
    den = sum(abs(s) for _, s in common)
    return float(num / den), len(common)


def make_itemcf_scorer(R, S_k, item_mean):
    Rb = R.sign()
    S_kt = S_k.T.tocsr()

    def score_fn(user_indices):
        num = np.asarray((R[user_indices] @ S_kt).todense(), dtype=np.float32)
        den = np.asarray((Rb[user_indices] @ S_kt).todense(), dtype=np.float32)
        out = np.where(den > 0, num / np.maximum(den, 1e-9), 0.0)
        fallback = den <= 0
        if fallback.any():
            out[fallback] = np.broadcast_to(item_mean, out.shape)[fallback]
        return out.astype(np.float32)

    return score_fn


def fit_svd(train_df, user_ids, book_ids, n_factors, n_epochs, seed):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(train_df[["user_id", "book_id", "rating"]], reader)
    trainset = data.build_full_trainset()
    algo = SurpriseSVD(n_factors=n_factors, n_epochs=n_epochs, random_state=seed)
    algo.fit(trainset)
    raw2u = {trainset.to_raw_uid(iu): iu for iu in trainset.all_users()}
    raw2i = {trainset.to_raw_iid(ii): ii for ii in trainset.all_items()}
    n_u, n_b = len(user_ids), len(book_ids)
    u_inner = np.array([raw2u.get(int(u), -1) for u in user_ids])
    i_inner = np.array([raw2i.get(int(b), -1) for b in book_ids])
    bu = np.zeros(n_u, np.float32)
    pu = np.zeros((n_u, n_factors), np.float32)
    mu = u_inner >= 0
    bu[mu] = algo.bu[u_inner[mu]]
    pu[mu] = algo.pu[u_inner[mu]]
    bi = np.zeros(n_b, np.float32)
    qi = np.zeros((n_b, n_factors), np.float32)
    mi = i_inner >= 0
    bi[mi] = algo.bi[i_inner[mi]]
    qi[mi] = algo.qi[i_inner[mi]]
    return {
        "mu": np.float32(trainset.global_mean),
        "bu": bu,
        "bi": bi,
        "pu": pu,
        "qi": qi,
    }


def make_svd_scorer(bundle):
    mu, bu, bi, pu, qi = (
        bundle["mu"],
        bundle["bu"],
        bundle["bi"],
        bundle["pu"],
        bundle["qi"],
    )

    def score_fn(user_indices):
        return (
            mu + bu[user_indices, None] + bi[None, :] + pu[user_indices] @ qi.T
        ).astype(np.float32)

    return score_fn


def predict_pairs(score_fn, user_idx, book_idx, chunk):
    order = np.argsort(user_idx, kind="stable")
    us = user_idx[order]
    out = np.empty(len(user_idx), np.float32)
    uniq = np.unique(us)
    for st in range(0, len(uniq), chunk):
        part = uniq[st : st + chunk]
        scores = score_fn(part)
        lo = np.searchsorted(us, part[0])
        hi = np.searchsorted(us, part[-1], "right")
        sel = order[lo:hi]
        local = np.searchsorted(part, user_idx[sel])
        out[sel] = scores[local, book_idx[sel]]
    return out


def z_norm(x):
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    return (x - mu) / (sd + 1e-9)


def z_norm_row(x):
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-9)


def make_hybrid_scorer(score_a, score_b, prior, w_a=0.5, w_prior=0.05):
    def score_fn(user_indices):
        a = z_norm(np.asarray(score_a(user_indices), np.float32))
        b = z_norm(np.asarray(score_b(user_indices), np.float32))
        c = z_norm(np.broadcast_to(prior, a.shape).astype(np.float32))
        return (w_a * a + (1 - w_a - w_prior) * b + w_prior * c).astype(np.float32)

    return score_fn


def build_pool(users, score_fns, seen_csr, n_books, top_per_model, pad, chunk):
    names = list(score_fns)
    pool_books = np.full((len(users), pad), -1, np.int32)
    pool_z = np.zeros((len(users), pad, len(names)), np.float32)
    for st in range(0, len(users), chunk):
        idx = users[st : st + chunk]
        seen = seen_csr[idx].toarray().astype(bool)
        mask = np.zeros((len(idx), n_books), bool)
        zs = []
        for fn in score_fns.values():
            s = np.asarray(fn(idx), np.float32)
            z = z_norm_row(s)
            top = topn_rows(np.where(seen, -np.inf, z), top_per_model)
            mask[np.arange(len(idx))[:, None], top] = True
            zs.append(z)
        for r in range(len(idx)):
            cols = np.flatnonzero(mask[r])
            pool_books[st + r, : len(cols)] = cols
            for mi, z in enumerate(zs):
                pool_z[st + r, : len(cols), mi] = z[r, cols]
    return pool_books, pool_z


def fast_metrics(users, pool_books, scores, rel, n_recs, ks=(5, 10)):
    disc = 1.0 / np.log2(np.arange(2, n_recs + 2))
    disc_cum = np.concatenate([[0.0], np.cumsum(disc)])
    part = np.argpartition(
        -np.where(pool_books < 0, -np.inf, scores), n_recs - 1, axis=1
    )[:, :n_recs]
    order = np.argsort(
        np.take_along_axis(-np.where(pool_books < 0, -np.inf, scores), part, axis=1),
        axis=1,
        kind="stable",
    )
    top = np.take_along_axis(part, order, axis=1)
    cand = np.take_along_axis(pool_books, top, axis=1)
    hits = np.asarray(rel[np.repeat(users, n_recs), cand.ravel()]).ravel()
    hits = hits.astype(np.float64).reshape(len(users), n_recs)
    n_rel = np.asarray(rel.sum(1)).ravel()[users]
    out = {}
    for k in ks:
        hk = hits[:, :k]
        out[f"P@{k}"] = float((hk.sum(1) / k).mean())
        out[f"R@{k}"] = float(
            np.divide(
                hk.sum(1), n_rel, out=np.zeros_like(n_rel), where=n_rel > 0
            ).mean()
        )
        dcg = (hits[:, :k] * disc[:k]).sum(1)
        idcg = disc_cum[np.clip(n_rel, 0, k).astype(int)]
        out[f"nDCG@{k}"] = float(
            np.divide(dcg, idcg, out=np.zeros_like(dcg), where=idcg > 0).mean()
        )
    return out


def weight_combos(step):
    steps = np.arange(0, 1.0 + step / 2, step)
    return [w for w in itertools.product(steps, repeat=4) if abs(sum(w) - 1) < 1e-9]
