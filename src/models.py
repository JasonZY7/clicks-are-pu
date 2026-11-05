"""One pointwise MF trainer (the families differ in their weights, nnPU in its step), BPR-MF, EASE and popularity."""
import time

import numpy as np
import torch

from config import BPR, DIM, EVERY, LR, RULES, TORCH_THREADS
from data import theta
from evaluate import traj_stats

torch.set_num_threads(TORCH_THREADS)


def weights(Y, n_i, family, c=None, eta=None, M=0.0):
    """(w1, w0) of J = sum w1 (1 - sigma)^2 + w0 sigma^2 for binary labels Y (any leading batch dims)."""
    if family == "naive":
        return Y, 1 - Y
    if family == "wmf":
        return c * Y, 1 - Y
    if family == "scar":
        return Y / c, 1 - Y / c
    th = theta(n_i, eta, M)
    w = (Y * np.where(th > 0, 1 / np.maximum(th, 1e-12), 0.0)).astype(Y.dtype)  # 0/0 cannot occur on Y = 1
    if family == "balanced":  # expected objective of the official code's 50/50 minibatch sampling
        N_pos = float(Y.sum())
        a, b = Y.size / (2 * (Y.size - N_pos)), Y.size / (2 * N_pos)
        return b * w, a * (1 - Y) + b * (Y - w)
    return w, 1 - w


def objective(S, w1, w0, lam, U, V, nnpu=False):
    """Returns (objective to descend, R_p, R_n, reg). nnPU is Kiryo et al. Algorithm 1 with beta = 0, gamma = 1."""
    sig = torch.sigmoid(S)
    Rp, Rn = (w1 * (1 - sig) ** 2).sum(), (w0 * sig**2).sum()
    reg = lam / 2 * (U.pow(2).sum() + V.pow(2).sum())
    return (-Rn + reg if nnpu and Rn < 0 else Rp + Rn + reg), Rp, Rn, reg


def _init(m, n, dim, seed):
    torch.manual_seed(seed)
    U, V = torch.nn.Parameter(torch.empty(m, dim)), torch.nn.Parameter(torch.empty(n, dim))
    torch.nn.init.xavier_uniform_(U)
    torch.nn.init.xavier_uniform_(V)
    return U, V


class Tracker:
    """Keeps random-set scores at each rule's best step; first maximum wins, same as evaluate.select."""

    def __init__(self, d, rules):
        self.d, self.rules, self.best, self.scores = d, rules, {}, {}

    def update(self, S, rec, last=False):
        u, i = self.d.rand[:, 0], self.d.rand[:, 1]
        for rule in self.rules:
            if rec[RULES[rule]] > self.best.get(rule, -np.inf):
                self.best[rule], self.scores[rule] = rec[RULES[rule]], S[u, i].copy()
        if last:
            self.scores.update({f"last_{rule}": S[u, i].copy() for rule in self.rules})


def fit_pointwise(d, cfg, seed, steps, track=()):
    """Full-batch Adam on the exact objective. Row at step t = parameters after t updates."""
    Y = d.Y
    w1, w0 = weights(Y, d.n_i, cfg["family"], cfg["c"], cfg["eta"], cfg["M"])
    lam = cfg["lam_rel"] * float(w1[Y == 1].mean())
    w1, w0 = torch.from_numpy(w1), torch.from_numpy(w0)
    U, V = _init(d.m, d.n, cfg["d"], seed)
    opt = torch.optim.Adam([U, V], lr=LR)
    fu, fi = torch.from_numpy(d.fit[:, 0]), torch.from_numpy(d.fit[:, 1])
    nnpu = cfg["family"] == "nnpu"
    tracker, rows, t0 = Tracker(d, track), [], time.perf_counter()
    for step in range(steps + 1):
        S = U @ V.T
        obj, Rp, Rn, reg = objective(S, w1, w0, lam, U, V, nnpu)
        if step % EVERY == 0:
            snap = S.detach()
            rec = dict(step=step, J=(Rp + Rn + reg).item(), R_n=Rn.item() / Y.size,
                       frac_pos=float((snap[fu, fi] > 0).float().mean()), **traj_stats(snap.numpy(), d))
            tracker.update(snap.numpy(), rec, last=step == steps)
            rec["wall"] = time.perf_counter() - t0
            rows.append(rec)
        if step < steps:
            opt.zero_grad()
            obj.backward()
            opt.step()
    return rows, tracker.scores


def fit_bpr(d, cfg, seed, track=()):
    """BPR-MF with uniform negatives, logsigmoid loss and L2. Row at step t = after t epochs."""
    reg = cfg["lam_rel"]
    U, V = _init(d.m, d.n, cfg["d"], seed)
    opt = torch.optim.Adam([U, V], lr=BPR["lr"])
    fu, fi = torch.from_numpy(d.fit[:, 0]), torch.from_numpy(d.fit[:, 1])
    tracker, rows, t0 = Tracker(d, track), [], time.perf_counter()
    for epoch in range(BPR["epochs"] + 1):
        with torch.no_grad():
            snap = U @ V.T
            rec = dict(step=epoch, J=np.nan, R_n=np.nan, frac_pos=float((snap[fu, fi] > 0).float().mean()),
                       **traj_stats(snap.numpy(), d))
            tracker.update(snap.numpy(), rec, last=epoch == BPR["epochs"])
            rec["wall"] = time.perf_counter() - t0
            rows.append(rec)
        if epoch == BPR["epochs"]:
            break
        idx = torch.randperm(len(fu))
        for s in range(0, len(idx), BPR["batch"]):
            b = idx[s:s + BPR["batch"]]
            uu, ii, jj = fu[b], fi[b], torch.randint(0, d.n, (len(b),))
            x = (U[uu] * (V[ii] - V[jj])).sum(1)
            l2 = (U[uu].pow(2).sum(1) + V[ii].pow(2).sum(1) + V[jj].pow(2).sum(1)).mean() / 2
            loss = -torch.nn.functional.logsigmoid(x).mean() + reg * l2
            opt.zero_grad()
            loss.backward()
            opt.step()
    return rows, tracker.scores


def ease_scores(Y, lam):
    """EASE closed form (Steck 2019): B = I - P diag(1/diag P) with zero diagonal, scores = X B."""
    X = Y.astype(np.float64)
    P = np.linalg.inv(X.T @ X + lam * np.eye(X.shape[1]))
    B = -P / np.diag(P)[None, :]
    np.fill_diagonal(B, 0)
    return (X @ B).astype(np.float32)


def fit_static(d, S, track=()):
    """A model without a trajectory: one row at step 0."""
    t0 = time.perf_counter()
    rec = dict(step=0, J=np.nan, R_n=np.nan, frac_pos=np.nan, **traj_stats(S, d))
    tracker = Tracker(d, track)
    tracker.update(S, rec, last=True)
    rec["wall"] = time.perf_counter() - t0
    return [rec], tracker.scores


def fit(d, cfg, seed, steps, track=()):
    family = cfg["family"]
    if family == "popularity":
        return fit_static(d, np.tile(d.n_i.astype(np.float32), (d.m, 1)), track)
    if family == "ease":
        return fit_static(d, ease_scores(d.Y, cfg["lam_rel"]), track)
    if family == "bpr":
        return fit_bpr(d, cfg, seed, track)
    return fit_pointwise(d, cfg, seed, steps, track)
