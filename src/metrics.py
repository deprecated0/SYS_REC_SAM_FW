"""Метрики ранжирования и сборка топ-N рекомендаций."""

import numpy as np
from scipy.sparse import csr_matrix


def precision_at_k(recommended, relevant, k):
    rec = np.asarray(recommended)[:k]
    return float(np.isin(rec, list(relevant)).mean())


def recall_at_k(recommended, relevant, k):
    rec = np.asarray(recommended)[:k]
    return float(np.isin(np.asarray(list(relevant)), rec).mean())


def ndcg_at_k(recommended, relevant, k):
    rec = np.asarray(recommended)[:k]
    hits = np.isin(rec, list(relevant)).astype(float)
    dcg = float((hits / np.log2(np.arange(2, len(rec) + 2))).sum())
    ideal_hits = min(len(relevant), k)
    idcg = float((1.0 / np.log2(np.arange(2, ideal_hits + 2))).sum())
    return dcg / idcg if ideal_hits > 0 else 0.0


def topn_rows(scores, n):
    part = np.argpartition(-scores, n - 1, axis=1)[:, :n]
    order = np.argsort(np.take_along_axis(-scores, part, axis=1), axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1)


def build_recs(score_fn, seen_csr, n, chunk, subset=None):
    subset = np.arange(seen_csr.shape[0]) if subset is None else subset
    recs = np.empty((len(subset), n), dtype=np.int32)
    for st in range(0, len(subset), chunk):
        idx = subset[st : st + chunk]
        scores = np.asarray(score_fn(idx), dtype=np.float32)
        seen_mask = seen_csr[idx].toarray().astype(bool)
        scores[seen_mask] = -np.inf
        recs[st : st + chunk] = topn_rows(scores, n)
    return recs


def evaluate(name, recs_matrix, rel_sets, users):
    res = {"model": name}
    for k in (5, 10):
        p = np.mean(
            [
                precision_at_k(recs_matrix[i], rel_sets[u], k)
                for i, u in enumerate(users)
            ]
        )
        r = np.mean(
            [recall_at_k(recs_matrix[i], rel_sets[u], k) for i, u in enumerate(users)]
        )
        nd = np.mean(
            [ndcg_at_k(recs_matrix[i], rel_sets[u], k) for i, u in enumerate(users)]
        )
        res.update({f"P@{k}": p, f"R@{k}": r, f"nDCG@{k}": nd})
    return res


def rel_info(test, n_users, n_books, threshold):
    rel_mask = (test.rating >= threshold).values
    test_rel = csr_matrix(
        (
            np.ones(rel_mask.sum(), np.float32),
            (test.user_idx.values[rel_mask], test.book_idx.values[rel_mask]),
        ),
        shape=(n_users, n_books),
    )
    rel_sets = {
        u: set(test_rel.getrow(u).indices)
        for u in np.unique(test.user_idx.values[rel_mask])
    }
    return rel_sets, np.array(sorted(rel_sets)), test_rel


def fast_metrics(users, pool_books, scores, rel, n_recs, ks=(5, 10)):
    """Быстрые метрики по готовому пулу кандидатов (векторизованно)."""
    disc = 1.0 / np.log2(np.arange(2, n_recs + 2))
    disc_cum = np.concatenate([[0.0], np.cumsum(disc)])
    masked = np.where(pool_books < 0, -np.inf, scores)
    part = np.argpartition(-masked, n_recs - 1, axis=1)[:, :n_recs]
    order = np.argsort(-np.take_along_axis(masked, part, axis=1), axis=1, kind="stable")
    top = np.take_along_axis(part, order, axis=1)
    cand = np.take_along_axis(pool_books, top, axis=1)
    hits = (
        np.asarray(rel[np.repeat(users, n_recs), cand.ravel()])
        .ravel()
        .astype(np.float64)
        .reshape(len(users), n_recs)
    )
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
