"""Стадия 6. Базовые рекомендации и метрики ранжирования на тесте."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from .config import (
    FIGURES_DIR,
    MODELS_DIR,
    SPLIT_DIR,
    TABLES_DIR,
    ensure_dirs,
    load_params,
)
from .io_utils import load_index_maps, save_fig, setup_plots
from .metrics import build_recs, evaluate, rel_info
from .models import make_content_scorer, make_itemcf_scorer, make_svd_scorer


def main():
    params = load_params()
    ensure_dirs()
    setup_plots()
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

    rng = np.random.default_rng(params["random_state"])

    def random_score(idx):
        return rng.random((len(idx), n_books), dtype=np.float32)

    def pop_score(idx):
        return np.broadcast_to(pop_scores, (len(idx), n_books)).copy()

    content_score = make_content_scorer(fav_train, item_profiles, pop_scores)
    itemcf_score = make_itemcf_scorer(R_train, S_topk, item_mean)
    svd_score = make_svd_scorer(svd_bundle)

    def svd_pq_score(idx):
        return svd_bundle["pu"][idx] @ svd_bundle["qi"].T

    rel_sets, eval_users, _ = rel_info(
        test, n_users, n_books, params["relevance_threshold"]
    )
    print(f"Пользователей с релевантным тестом: {len(eval_users):,} из {n_users:,}")
    R_seen = R_train.sign().astype(np.bool_)
    n_recs, chunk = params["n_recs"], params["chunk"]
    recs = {
        "Random": build_recs(random_score, R_seen, n_recs, chunk),
        "Popularity": build_recs(pop_score, R_seen, n_recs, chunk),
        "Content (TF-IDF)": build_recs(content_score, R_seen, n_recs, chunk),
        f"ItemKNN CF (cos, K={params['top_k_neighbours']})": build_recs(
            itemcf_score, R_seen, n_recs, chunk
        ),
        f"SVD (surprise, d={params['svd']['n_factors']})": build_recs(
            svd_score, R_seen, n_recs, chunk
        ),
        "SVD (только p·q)": build_recs(svd_pq_score, R_seen, n_recs, chunk),
    }
    results = pd.DataFrame(
        [evaluate(name, recs[name][eval_users], rel_sets, eval_users) for name in recs]
    ).set_index("model")
    results.to_csv(TABLES_DIR / "metrics_main.csv", index_label="model")
    print(results)
    ax = results[["P@10", "R@10", "nDCG@10"]].plot(
        kind="bar", figsize=(10, 4.5), rot=15, colormap="crest"
    )
    ax.set_title("Базовые модели: метрики top-10 рекомендаций")
    ax.set_xlabel("Модель")
    ax.set_ylabel("Значение метрики")
    ax.legend(loc="upper right", ncol=3)
    ax.figure.tight_layout()
    save_fig(ax.figure, "metrics_bar.png")


if __name__ == "__main__":
    main()
