"""Стадия 4. Коллаборативная фильтрация: Item-Based CF."""

import time

import numpy as np
import pandas as pd
from scipy.sparse import save_npz

from .config import MODELS_DIR, SPLIT_DIR, TABLES_DIR, ensure_dirs, load_params
from .io_utils import build_interaction_matrices, load_index_maps
from .models import item_similarity, predict_rating_itemcf


def main():
    params = load_params()
    ensure_dirs()
    train = pd.read_csv(SPLIT_DIR / "train.csv")
    test = pd.read_csv(SPLIT_DIR / "test.csv")
    user_ids, book_ids = load_index_maps(SPLIT_DIR / "index_maps.npz")
    n_users, n_books = len(user_ids), len(book_ids)
    R_train, _ = build_interaction_matrices(
        train, n_users, n_books, params["relevance_threshold"]
    )
    density = R_train.nnz / (n_users * n_books) * 100
    print(f"R_train: {R_train.shape}, nnz={R_train.nnz:,} (плотность {density:.3f}%)")
    save_npz(MODELS_DIR / "R_train.npz", R_train)
    t0 = time.perf_counter()
    S_topk = item_similarity(
        R_train, lmbda=params["shrinkage_lambda"], k=params["top_k_neighbours"]
    )
    print(f"S_topk за {time.perf_counter() - t0:.1f} с, nnz={S_topk.nnz:,}")
    save_npz(MODELS_DIR / "S_topk.npz", S_topk)
    glob_mean = float(train.rating.mean())
    item_mean = np.full(n_books, glob_mean, dtype=np.float32)
    grp = train.groupby("book_idx")["rating"].mean()
    item_mean[grp.index] = grp.values.astype(np.float32)
    np.save(MODELS_DIR / "item_mean.npy", item_mean)
    u_hist = train.user_idx.value_counts()
    sample = test[test.user_idx.isin(u_hist[u_hist >= 5].index)].sample(
        3, random_state=params["random_state"]
    )
    rows = []
    for _, row in sample.iterrows():
        pred, n_used = predict_rating_itemcf(
            int(row.user_idx),
            int(row.book_idx),
            S_topk,
            R_train,
            item_mean,
            params["top_k_neighbours"],
        )
        rows.append(
            {
                "user_id": int(row.user_id),
                "book_id": int(row.book_id),
                "fact": int(row.rating),
                "pred": round(pred, 2),
                "n_used": n_used,
            }
        )
    pd.DataFrame(rows).to_csv(TABLES_DIR / "itemcf_demo.csv", index=False)


if __name__ == "__main__":
    main()
