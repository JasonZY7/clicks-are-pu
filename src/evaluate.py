"""Per-user ranking metrics, selection statistics, floors, paired bootstrap and calibration. numpy + pandas."""
import numpy as np
import pandas as pd

from config import BOOTSTRAP_SEED, PROC, RULES, TRAJ

take = np.take_along_axis


def orders(s, M):
    """Candidate positions by descending score. Stable, so ties go to the lower item id."""
    return np.argsort(-np.where(M, s, -np.inf), axis=1, kind="stable")


def discount(k, official=False):
    """Paper: 1/log2(rank+1). Official code: 1/log2(max(rank, 2)), so ranks 1 and 2 weigh the same."""
    r = np.arange(1, k + 1)
    return 1 / np.log2(np.maximum(r, 2)) if official else 1 / np.log2(r + 1)


def dcg(o, rel, K, official=False):
    r = take(rel, o[:, :K], 1)
    return (r * discount(r.shape[1], official)).sum(1)


def recall(o, rel, K):
    return take(rel, o[:, :K], 1).sum(1) / rel.sum(1)


def ap(o, rel, K, normalise=True):
    """Standard AP@K. normalise=False is the official code's sum, which can exceed 1."""
    r = take(rel, o[:, :K], 1)
    s = (r.cumsum(1) / np.arange(1, r.shape[1] + 1) * r).sum(1)
    return s / np.minimum(K, rel.sum(1)) if normalise else s


def per_user(s, L, rel=None, mask=None, Ks=(1, 3, 5)):
    """Metric arrays over users with at least one relevant candidate. rel and mask give the rare readings."""
    rel, mask = (L.R if rel is None else rel), (L.M if mask is None else mask)
    keep = rel.sum(1) > 0
    o, r = orders(s[keep], mask[keep]), rel[keep]
    out = {}
    for K in Ks:
        out[f"DCG@{K}"], out[f"Recall@{K}"], out[f"AP@{K}"] = dcg(o, r, K), recall(o, r, K), ap(o, r, K)
    return out


def floors(n_rel, L, Ks=(1, 3, 5)):
    """Analytic (random, ideal) metrics per user for n_rel relevant items among L candidates."""
    n, L = np.asarray(n_rel, float), np.asarray(L, float)
    random, ideal = {}, {}
    for K in Ks:
        rank = np.arange(1, K + 1)[:, None]
        d = (1 / np.log2(rank + 1))
        random[f"DCG@{K}"] = ((rank <= L) * d * n / L).sum(0)
        random[f"Recall@{K}"] = np.minimum(K, L) / L
        random[f"AP@{K}"] = ((rank <= L) * n / L / rank * (1 + (rank - 1) * (n - 1) / np.maximum(L - 1, 1))).sum(0) / np.minimum(K, n)
        ideal[f"DCG@{K}"] = ((rank <= n) * d).sum(0)
        ideal[f"Recall@{K}"] = np.minimum(K, n) / n
        ideal[f"AP@{K}"] = np.ones_like(n)
    return random, ideal


def snips(o, W, K=5):
    """(per-user, pooled) self-normalised DCG@K with weights W. o holds the full ranking."""
    w = take(W, o[:, :K], 1) * discount(K)
    return w.sum(1) / W.sum(1), w.sum() / W.sum()


def traj_stats(S, d):
    """Every column logged along a trajectory, from a dense score matrix S."""
    va, vb, ts = d.val_a, d.val_b, d.test
    o = orders(S[va.users[:, None], va.I], va.M)
    per, pooled = snips(o, va.W)
    out = {"val_a_snips": per.mean(), "val_a_pooled": pooled, "val_a_naive": snips(o, va.R)[0].mean()}
    v = dcg(orders(S[vb.users[:, None], vb.I], vb.M), vb.R, 5)
    out["val_b"] = v[vb.R.sum(1) > 0].mean()
    out.update({k: v[mask].mean() for k, mask in d.slices.items()})
    o = orders(S[ts.users[:, None], ts.I], ts.M)
    rel_rare = ts.R * d.rare[ts.I]
    out["test_dcg5"] = dcg(o, ts.R, 5)[ts.R.sum(1) > 0].mean()
    out["rare_credit5"] = dcg(o, rel_rare, 5)[rel_rare.sum(1) > 0].mean()
    return {k: float(v) for k, v in out.items()}


def load_traj(dataset=None):
    """Trajectories in a fixed order, so ties in select() do not depend on which fit finished first."""
    df = pd.read_csv(TRAJ, engine="pyarrow")  # exact float parsing: finals must reuse the exact c and lambda
    if dataset:
        df = df[df.dataset == dataset]
    fit = ["dataset", "family", "c", "eta", "M", "lam_rel", "d", "seed", "phase", "step"]
    return df.drop_duplicates(fit).sort_values(fit, kind="stable")  # a re-run fit repeats its earlier rows exactly, wall aside


def select(df, family, rule, phase="grid", seed=None):
    """The (config, step) row of a family that maximises a rule's statistic. First maximum wins ties."""
    rows = df[(df.family == family) & (df.phase == phase)]
    if seed is not None:
        rows = rows[rows.seed == seed]
    return rows.loc[rows[RULES[rule]].idxmax()]


def boot_idx(n, B=1000):
    """User-level resamples, shared by every comparison over the same n users."""
    return np.random.default_rng(BOOTSTRAP_SEED).integers(0, n, (B, n))


def paired_ci(a, b, idx):
    """Mean difference a - b and its 95% percentile bootstrap CI."""
    diff = np.asarray(a) - np.asarray(b)
    lo, hi = np.percentile(diff[idx].mean(1), [2.5, 97.5])
    return diff.mean(), lo, hi


def load_scores(dataset):
    return pd.read_parquet(PROC / f"final_scores_{dataset}.parquet")


def score_vectors(scores, family, rule):
    """{seed: score of every random-set row} for one family's final fits under one rule."""
    s = scores[(scores.family == family) & (scores.rule == rule)].sort_values("row_id", kind="stable")
    return {seed: g.score.to_numpy() for seed, g in s.groupby("seed")}


def seed_metrics(vectors, L, rel=None, mask=None, Ks=(1, 3, 5)):
    """(per-user metrics averaged over seeds, per-user metrics of each seed)."""
    each = [per_user(v[L.row], L, rel, mask, Ks) for v in vectors.values()]
    return {k: np.mean([e[k] for e in each], axis=0) for k in each[0]}, each


def compare(a, b, metric, idx, ideal):
    """Paired comparison of two seed_metrics results: CI, relative gain, headroom share, seed agreement."""
    diff, lo, hi = paired_ci(a[0][metric], b[0][metric], idx)
    base = b[0][metric].mean()
    n = max(len(a[1]), len(b[1]))  # a deterministic model has one seed: compare it with every seed of the other
    per_seed = [a[1][i % len(a[1])][metric].mean() - b[1][i % len(b[1])][metric].mean() for i in range(n)]
    return dict(model=a[0][metric].mean(), baseline=base, diff=diff, lo=lo, hi=hi, rel_gain=diff / base,
                headroom=diff / (ideal - base), same_sign=len({np.sign(v) for v in per_seed}) == 1)


def verdict(row):
    """The wording fixed before the grid for a sign test: CI above zero and all seeds agree, else a weaker word."""
    if row["lo"] > 0 and row["same_sign"]:
        return "held"
    return "same direction, not significant" if row["diff"] > 0 else "did not hold"


def reliability(p, y, bins=10):
    """Equal-mass bins of predicted probability against the observed rate, and the ECE."""
    o = np.argsort(p, kind="stable")
    parts = np.array_split(o, bins)
    tab = pd.DataFrame({"pred": [p[x].mean() for x in parts], "obs": [y[x].mean() for x in parts],
                        "n": [len(x) for x in parts]})
    return tab, float((tab.n / tab.n.sum() * (tab.pred - tab.obs).abs()).sum())
