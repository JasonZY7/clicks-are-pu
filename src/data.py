"""Parse, split and index the two datasets. numpy only, so the notebooks never import torch."""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from config import RAW, SNIPS_FLOOR, SPLIT_SEED, THRESHOLD, USER_SPLIT_SEED, VAL_NEGATIVES

SHAPE = {"yahoo": (15400, 1000), "coat": (290, 300)}
# (train ratings, train positives, random-set ratings, random-set positives), measured on the downloaded files. The positive counts are not published figures.
COUNTS = {"yahoo": (311704, 125077, 54000, 4749), "coat": (6960, 1905, 4640, 860)}


def theta(n_i, eta, M=0.0):
    """Eq. 12 with Eq. 10-11 clipping: (n_i / max n)^eta, floored at M. Zero for unseen items when M = 0."""
    return np.maximum((n_i / n_i.max()) ** eta, M)


@dataclass
class Lists:
    """Per-user candidate lists padded to a rectangle (K users x L slots), sorted by item id in a row."""
    users: np.ndarray
    I: np.ndarray  # item ids
    R: np.ndarray  # relevance
    M: np.ndarray  # True on real candidates
    W: np.ndarray | None = None  # SNIPS weights
    row: np.ndarray | None = None  # row ids into Data.rand, -1 on padding


def make_lists(u, i, rel, w=None, row=None):
    o = np.lexsort((i, u))
    users, inv, cnt = np.unique(u[o], return_inverse=True, return_counts=True)
    slot = np.arange(len(u)) - np.r_[0, cnt.cumsum()[:-1]][inv]

    def fill(x, dtype, pad=0):
        a = np.full((len(users), cnt.max()), pad, dtype)
        a[inv, slot] = x
        return a

    return Lists(users, fill(i[o], np.int64), fill(rel[o], np.float32), fill(True, bool),
                 None if w is None else fill(w[o], np.float32),
                 None if row is None else fill(row[o], np.int64, -1))


@dataclass
class Data:
    name: str
    m: int
    n: int
    train: np.ndarray  # all self-selected ratings (user, item, rating)
    rand: np.ndarray  # all uniform-exposure ratings (user, item, rating)
    split: np.ndarray  # per rand row: "val_b", "test" or "dropped" (also rated in train)
    fit: np.ndarray  # training positives
    val: np.ndarray  # held-out 10% of the positives
    n_i: np.ndarray  # training positives per item
    rare: np.ndarray  # bottom 50% of items by n_i, ties by item id
    Y: np.ndarray  # dense training labels, float32
    val_a: Lists
    val_b: Lists
    test: Lists
    slices: dict  # sub-slice name -> mask over val_b users with at least one relevant item


def _read_raw(name):
    if name == "yahoo":
        train = np.loadtxt(RAW / "yahoo" / "user.txt", delimiter=",", dtype=np.int64)
        rand = np.loadtxt(RAW / "yahoo" / "random.txt", delimiter=",", dtype=np.int64)
    else:
        def rows(path):
            a = np.loadtxt(path, dtype=np.int64)
            u, i = a.nonzero()
            return np.c_[u, i, a[u, i]]
        train, rand = rows(RAW / "coat" / "train.ascii"), rows(RAW / "coat" / "test.ascii")
    got = (len(train), (train[:, 2] >= THRESHOLD).sum(), len(rand), (rand[:, 2] >= THRESHOLD).sum())
    assert got == COUNTS[name], f"{name}: got {got}, expected {COUNTS[name]}"
    return train, rand


def _val_candidates(fit, val, m, n, n_i):
    """Each validation user's positives plus 100 uniform unlabelled items, sampled once."""
    rng = np.random.default_rng(SPLIT_SEED)
    known = np.zeros((m, n), bool)
    known[fit[:, 0], fit[:, 1]] = known[val[:, 0], val[:, 1]] = True
    neg = [(u, rng.choice(np.flatnonzero(~known[u]), VAL_NEGATIVES, replace=False)) for u in np.unique(val[:, 0])]
    u = np.r_[val[:, 0], np.repeat([a for a, _ in neg], VAL_NEGATIVES)]
    i = np.r_[val[:, 1], np.concatenate([b for _, b in neg])]
    rel = np.r_[np.ones(len(val)), np.zeros(len(u) - len(val))]
    w = rel / np.maximum(theta(n_i, 0.5), SNIPS_FLOOR)[i]
    return make_lists(u, i, rel, w)


@lru_cache(maxsize=None)
def load(name):
    m, n = SHAPE[name]
    train, rand = _read_raw(name)
    pos = train[train[:, 2] >= THRESHOLD]
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(pos))
    k = int(0.1 * len(pos))
    val, fit = pos[perm[:k]], pos[perm[k:]]
    n_i = np.bincount(fit[:, 1], minlength=n).astype(np.float64)
    rare = np.zeros(n, bool)
    rare[np.lexsort((np.arange(n), n_i))[:n // 2]] = True
    Y = np.zeros((m, n), np.float32)
    Y[fit[:, 0], fit[:, 1]] = 1
    assert Y.sum() == len(fit), "duplicate training pairs"

    # Random set split by user. Coat cells also rated in train leave both slices (see NB1).
    rated = np.zeros((m, n), bool)
    if name == "coat":
        rated[train[:, 0], train[:, 1]] = True
    users = np.random.default_rng(USER_SPLIT_SEED).permutation(np.unique(rand[:, 0]))
    nb = int(0.1 * len(users))
    split = np.where(np.isin(rand[:, 0], users[:nb]), "val_b", "test").astype(object)
    split[rated[rand[:, 0], rand[:, 1]]] = "dropped"
    rel = (rand[:, 2] >= THRESHOLD).astype(np.float32)
    rows = np.arange(len(rand))

    def part(tag):
        s = split == tag
        return make_lists(rand[s, 0], rand[s, 1], rel[s], row=rows[s])

    val_b, test = part("val_b"), part("test")
    ev = val_b.R.sum(1) > 0
    q = np.array_split(users[:nb], 4)  # disjoint user slices, for the noise estimate in NB3
    groups = {"B5a": q[0:2], "B5b": q[2:4], "B2.5a": q[:1], "B2.5b": q[1:2], "B2.5c": q[2:3], "B2.5d": q[3:4]}
    slices = {k: np.isin(val_b.users, np.concatenate(v)) & ev for k, v in groups.items()}
    return Data(name, m, n, train, rand, split, fit, val, n_i, rare, Y,
                _val_candidates(fit, val, m, n, n_i), val_b, test, slices)
