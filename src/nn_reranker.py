"""Wide & Deep реранкер на PyTorch."""

import numpy as np
import torch
import torch.nn as nn


class WideDeep(nn.Module):
    def __init__(self, n_users, n_books, n_feats, emb=16, use_deep=True):
        super().__init__()
        self.use_deep = use_deep
        self.wide = nn.Linear(n_feats, 1)
        if use_deep:
            self.emb_u = nn.Embedding(n_users, emb)
            self.emb_b = nn.Embedding(n_books, emb)
            self.mlp = nn.Sequential(
                nn.Linear(2 * emb + n_feats, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

    def forward(self, feats, u, b):
        out = self.wide(feats)
        if self.use_deep:
            h = torch.cat([self.emb_u(u), self.emb_b(b), feats], dim=1)
            out = out + self.mlp(h)
        return out.squeeze(-1)


def train_reranker(
    X, u, b, y, n_users, n_books, n_epochs=3, batch=16384, lr=1e-3, use_deep=True
):
    model = WideDeep(n_users, n_books, X.shape[1], use_deep=use_deep)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    Xt = torch.from_numpy(X)
    ut = torch.from_numpy(u)
    bt = torch.from_numpy(b)
    yt = torch.from_numpy(y)
    n = len(y)
    for ep in range(n_epochs):
        perm = torch.randperm(n)
        ep_loss = 0.0
        for st in range(0, n, batch):
            sel = perm[st : st + batch]
            logits = model(Xt[sel], ut[sel], bt[sel])
            loss = loss_fn(logits, yt[sel])
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss) * len(sel)
        print(f"  epoch {ep + 1}: loss = {ep_loss / n:.4f}")
    return model


def nn_scores(model, feats, users_arr, books_arr, batch=200_000):
    model.eval()
    out = np.empty(len(users_arr), np.float32)
    with torch.no_grad():
        for st in range(0, len(users_arr), batch):
            sl = slice(st, st + batch)
            out[sl] = model(
                torch.from_numpy(feats[sl]),
                torch.from_numpy(users_arr[sl]),
                torch.from_numpy(books_arr[sl]),
            ).numpy()
    return out


def make_training_pairs(
    train_df, R_train, n_users, n_books, n_pos, n_neg, relevance_threshold, seed
):
    seen_rows = R_train.sign().tocoo()
    seen_sets = {}
    for u, b in zip(seen_rows.row, seen_rows.col):
        seen_sets.setdefault(u, []).append(b)
    hi = train_df[train_df.rating >= relevance_threshold].groupby("user_idx")[
        "book_idx"
    ]
    fav_lists = hi.apply(list)
    rng = np.random.default_rng(seed)
    users, books, labels = [], [], []
    for u in range(n_users):
        fav = fav_lists.get(u, [])
        if not fav:
            continue
        pos = rng.choice(fav, size=min(n_pos, len(fav)), replace=False)
        cand = rng.choice(n_books, size=n_neg + 25, replace=False)
        neg = cand[~np.isin(cand, seen_sets.get(u, []))][:n_neg]
        users += [u] * (len(pos) + len(neg))
        books += list(pos) + list(neg)
        labels += [1] * len(pos) + [0] * len(neg)
    return (
        np.array(users, np.int64),
        np.array(books, np.int64),
        np.array(labels, np.float32),
    )
