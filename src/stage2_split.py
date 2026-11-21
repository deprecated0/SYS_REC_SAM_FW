"""Стадия 2. Трёхстороннее разделение: train / val / test."""

import numpy as np
import pandas as pd

from .config import SPLIT_DIR, TABLES_DIR, ensure_dirs, load_params
from .io_utils import (
    add_indices,
    build_index_maps,
    load_raw_data,
    make_id2idx,
    save_index_maps,
    save_json,
)


def split_data(ratings_df, min_train, n_val, n_test, seed):
    counts = ratings_df.user_id.value_counts()
    eligible = counts.index[counts >= min_train]
    shuffled = (
        ratings_df[ratings_df.user_id.isin(eligible)]
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    shuffled["rank_in_user"] = shuffled.groupby("user_id").cumcount()
    val = shuffled[shuffled.rank_in_user < n_val].drop(columns="rank_in_user")
    test = shuffled[
        (shuffled.rank_in_user >= n_val) & (shuffled.rank_in_user < n_val + n_test)
    ].drop(columns="rank_in_user")
    train = pd.concat(
        [
            shuffled[shuffled.rank_in_user >= n_val + n_test].drop(
                columns="rank_in_user"
            ),
            ratings_df[~ratings_df.user_id.isin(eligible)],
        ],
        ignore_index=True,
    )
    return train, val, test


def main():
    params = load_params()
    ensure_dirs()
    ratings, _, _, _ = load_raw_data()
    train, val, test = split_data(
        ratings,
        params["min_train_ratings"],
        params["n_val"],
        params["n_test"],
        params["random_state"],
    )
    user_ids, book_ids = build_index_maps(ratings)
    uid2i = make_id2idx(user_ids)
    bid2i = make_id2idx(book_ids)
    for df in (train, val, test):
        df["user_idx"] = df.user_id.map(uid2i)
        df["book_idx"] = df.book_id.map(bid2i)
    train.to_csv(SPLIT_DIR / "train.csv", index=False)
    val.to_csv(SPLIT_DIR / "val.csv", index=False)
    test.to_csv(SPLIT_DIR / "test.csv", index=False)
    save_index_maps(SPLIT_DIR / "index_maps.npz", user_ids, book_ids)
    train_counts = (
        train.groupby("user_idx").size().reindex(np.arange(len(user_ids)), fill_value=0)
    )
    summary = {
        "train_size": int(len(train)),
        "val_size": int(len(val)),
        "test_size": int(len(test)),
        "val_users": int(val.user_id.nunique()),
        "test_users": int(test.user_id.nunique()),
        "total_users": int(len(user_ids)),
        "total_books": int(len(book_ids)),
        "users_lt5_train": int((train_counts < 5).sum()),
        "users_lt5_train_percent": round(float((train_counts < 5).mean() * 100), 1),
    }
    save_json(summary, TABLES_DIR / "split_summary.json")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
