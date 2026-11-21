"""Стадия 3. Популярная и контентная модели."""

import numpy as np
import pandas as pd
from scipy.sparse import save_npz

from .config import (
    DATA_DIR,
    MODELS_DIR,
    SPLIT_DIR,
    TABLES_DIR,
    ensure_dirs,
    load_params,
)
from .io_utils import load_index_maps, load_raw_data, make_id2idx, prepare_book_tags
from .models import (
    META_TAGS,
    build_book_profiles,
    build_favorites,
    find_demo_book,
    fit_item_profiles,
    get_similar_books,
    popularity_scores,
)


def main():
    params = load_params()
    ensure_dirs()
    train = pd.read_csv(SPLIT_DIR / "train.csv")
    user_ids, book_ids = load_index_maps(SPLIT_DIR / "index_maps.npz")
    n_users, n_books = len(user_ids), len(book_ids)
    bid2i = make_id2idx(book_ids)
    _, books, tags, book_tags = load_raw_data()
    bt = prepare_book_tags(books, book_tags, tags)
    books_by_id = books.set_index("id")

    pop_scores, pop_stats = popularity_scores(
        train, n_books, params["min_ratings_popular"]
    )
    np.save(MODELS_DIR / "pop_scores.npy", pop_scores)
    pop_stats.to_csv(TABLES_DIR / "pop_stats.csv", index_label="book_idx")
    print(
        f"Проходят порог {params['min_ratings_popular']} оценок: {(pop_scores > 0).sum()} из {n_books}"
    )

    pop_top = (
        pop_stats[pop_stats.n >= params["min_ratings_popular"]]
        .nlargest(10, "mean")
        .reset_index()
    )
    pop_top["book_id"] = book_ids[pop_top["book_idx"].values]
    pop_top["title"] = books_by_id.loc[pop_top["book_id"].values, "title"].values
    pop_top["authors"] = books_by_id.loc[pop_top["book_id"].values, "authors"].values
    pop_top[["book_id", "title", "authors", "n", "mean"]].to_csv(
        TABLES_DIR / "popular_top10_train.csv", index=False
    )

    profiles = build_book_profiles(books, bt, book_ids, bid2i, META_TAGS)
    item_profiles = fit_item_profiles(profiles, min_df=params["content"]["min_df"])
    save_npz(MODELS_DIR / "item_profiles.npz", item_profiles)
    print(f"Матрица профилей: {item_profiles.shape}, nnz={item_profiles.nnz}")

    demo_rows = []
    for pattern in ("Hunger Games", "Hobbit", "1984"):
        book_id = find_demo_book(books, pattern)
        idx = bid2i[book_id]
        top_idx, sims = get_similar_books(item_profiles, idx, n=5)
        source_title = books_by_id.loc[book_id, "title"]
        for rank, (j, s) in enumerate(zip(top_idx, sims), start=1):
            nb_id = int(book_ids[j])
            demo_rows.append(
                {
                    "source_book_id": book_id,
                    "source_title": source_title,
                    "rank": rank,
                    "book_id": nb_id,
                    "title": books_by_id.loc[nb_id, "title"],
                    "authors": books_by_id.loc[nb_id, "authors"],
                    "cosine": round(float(s), 3),
                }
            )
    pd.DataFrame(demo_rows).to_csv(TABLES_DIR / "similar_books_demo.csv", index=False)

    fav_train = build_favorites(train, n_users, n_books, params["relevance_threshold"])
    save_npz(MODELS_DIR / "fav_train.npz", fav_train)
    print(f"fav_train: {fav_train.shape}, nnz={fav_train.nnz}")


if __name__ == "__main__":
    main()
