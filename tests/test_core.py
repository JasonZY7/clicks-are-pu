"""Plain asserts on the parts a wrong sign would silently break. Run: make test"""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from data import make_lists, theta
from evaluate import ap, dcg, floors, orders, recall, snips
from models import objective, weights

D1 = lambda s: (1 - s) ** 2
D0 = lambda s: s**2


def test_relmf_is_unbiased_and_naive_is_not():
    """Enumerate all 2^18 outcomes of (O, R) on a 3 x 3 toy: E[Eq. 9] equals the ideal loss exactly."""
    rng = np.random.default_rng(0)
    n_i = np.array([1.0, 4.0, 9.0])
    th = np.tile(theta(n_i, 0.5), (3, 1))
    gamma, sig = rng.uniform(0.05, 0.95, (3, 3)), rng.uniform(0.05, 0.95, (3, 3))
    bits = ((np.arange(2**18)[:, None] >> np.arange(18)) & 1).astype(float)
    O, R = bits[:, :9].reshape(-1, 3, 3), bits[:, 9:].reshape(-1, 3, 3)
    p = (th**O * (1 - th) ** (1 - O) * gamma**R * (1 - gamma) ** (1 - R)).prod((1, 2))
    Y = O * R
    ideal = (gamma * D1(sig) + (1 - gamma) * D0(sig)).mean()

    def expected(family, **kw):
        w1, w0 = weights(Y, n_i, family, **kw)
        return ((w1 * D1(sig) + w0 * D0(sig)).mean((1, 2)) * p).sum()

    assert abs(p.sum() - 1) < 1e-12
    assert abs(expected("relmf", eta=0.5) - ideal) < 1e-12
    assert abs(expected("naive") - ideal) > 1e-3
    # The same check through the objective the trainer descends, which is linear in (w1, w0).
    w1, w0 = (torch.from_numpy((w * p[:, None, None]).sum(0)) for w in weights(Y, n_i, "relmf", eta=0.5))
    U = torch.eye(3, dtype=w1.dtype)
    _, Rp, Rn, reg = objective(torch.logit(torch.from_numpy(sig)), w1, w0, 0.5, U, U)
    assert abs((Rp + Rn).item() / 9 - ideal) < 1e-12 and reg == 1.5


def test_scar_is_wmf_plus_a_positive_sigma_term():
    rng = np.random.default_rng(1)
    Y, sig, c = rng.integers(0, 2, (20, 30)).astype(float), rng.uniform(0.01, 0.99, (20, 30)), 0.1
    w1, w0 = weights(Y, None, "scar", c=c)
    v1, v0 = weights(Y, None, "wmf", c=1 / c)
    J_scar, J_wmf = (w1 * D1(sig) + w0 * D0(sig)).sum(), (v1 * D1(sig) + v0 * D0(sig)).sum()
    assert abs(J_scar - J_wmf - ((1 - 1 / c) * Y * D0(sig)).sum()) < 1e-9


def test_balanced_is_the_expected_minibatch_objective():
    """Sum over a 50/50 positive/unlabelled batch, in expectation, with every pair rescaled to |D| terms."""
    rng = np.random.default_rng(2)
    Y = (rng.uniform(size=(30, 40)) < 0.1).astype(float)
    n_i = Y.sum(0) + 1
    sig = rng.uniform(0.01, 0.99, Y.shape)
    w, _ = weights(Y, n_i, "relmf", eta=0.5)
    pos, unl = Y == 1, Y == 0
    per_pair = w * D1(sig) + (1 - w) * D0(sig)
    expected = Y.size * (per_pair[pos].mean() + per_pair[unl].mean()) / 2
    a1, a0 = weights(Y, n_i, "balanced", eta=0.5)
    assert abs((a1 * D1(sig) + a0 * D0(sig)).sum() - expected) < 1e-9


def test_zero_positive_item_has_zero_weight():
    Y = np.array([[1.0, 0.0], [0.0, 0.0]])
    w1, w0 = weights(Y, np.array([1.0, 0.0]), "relmf", eta=0.5)
    assert np.isfinite(w1).all() and w1[0, 1] == 0 and w0[0, 1] == 1


def test_clipping_below_min_theta_is_a_no_op():
    n_i = np.array([1.0, 100.0])
    assert (theta(n_i, 0.5, 0.05) == theta(n_i, 0.5)).all()  # min theta is 0.1
    assert theta(n_i, 0.5, 0.3)[0] == 0.3


def test_metrics_on_hand_examples():
    # A single relevant item at rank 2 among 3.
    s = np.array([[3.0, 2.0, 1.0]])
    rel, M = np.array([[0.0, 1.0, 0.0]]), np.ones((1, 3), bool)
    o = orders(s, M)
    assert abs(dcg(o, rel, 3)[0] - 0.63093) < 1e-4
    assert abs(dcg(o, rel, 3, official=True)[0] - 1.0) < 1e-12
    assert recall(o, rel, 1)[0] == 0 and recall(o, rel, 2)[0] == 1
    # Relevant at ranks 1 and 3, K = 3: AP = (1/1 + 2/3) / min(3, 2).
    rel = np.array([[1.0, 0.0, 1.0]])
    assert abs(ap(o, rel, 3)[0] - (1 + 2 / 3) / 2) < 1e-12
    assert abs(ap(o, rel, 3, normalise=False)[0] - (1 + 2 / 3)) < 1e-12
    # K = 5 with one relevant item at rank 1 stays at 1, unlike the formula printed in the paper.
    assert ap(o, np.array([[1.0, 0.0, 0.0]]), 5)[0] == 1


def test_ties_go_to_the_lower_item_id_and_padding_ranks_last():
    L = make_lists(np.array([0, 0, 0, 1]), np.array([7, 2, 5, 3]), np.array([0.0, 1.0, 0.0, 1.0]))
    assert L.I.tolist() == [[2, 5, 7], [3, 0, 0]] and L.M[1].tolist() == [True, False, False]
    o = orders(np.zeros(L.I.shape), L.M)
    assert o[0].tolist() == [0, 1, 2] and o[1, 0] == 0
    assert orders(np.array([[-5.0, 9.0, 9.0]]), np.array([[True, False, False]]))[0, 0] == 0


def test_floors_match_enumeration():
    L, n_rel, K = 5, 2, 3
    rows = {"DCG": [], "Recall": [], "AP": []}
    for perm in itertools.permutations(range(L)):
        rel = np.zeros((1, L))
        rel[0, :n_rel] = 1
        o = np.array([perm])
        rows["DCG"].append(dcg(o, rel, K)[0])
        rows["Recall"].append(recall(o, rel, K)[0])
        rows["AP"].append(ap(o, rel, K)[0])
    random, ideal = floors([n_rel], [L], (K,))
    for m, v in rows.items():
        assert abs(random[f"{m}@{K}"][0] - np.mean(v)) < 1e-12, m
        assert abs(ideal[f"{m}@{K}"][0] - np.max(v)) < 1e-12, m


def test_snips():
    # With one positive per user the weight cancels: per-user SNIPS equals the unweighted statistic.
    o = np.array([[1, 0, 2], [0, 1, 2]])
    W = np.array([[0.0, 7.0, 0.0], [0.0, 3.0, 0.0]])
    unweighted = (W > 0).astype(float)
    per, pooled = snips(o, W, K=3)
    per0, _ = snips(o, unweighted, K=3)
    assert (per == per0).all()
    # Pooled keeps the weights across users, so it differs from the per-user mean.
    assert abs(pooled - per.mean()) > 1e-3


def test_nnpu_branch():
    U, V = torch.tensor([[1.0]], requires_grad=True), torch.tensor([[1.0]], requires_grad=True)
    S = U @ V.T
    w1, w0 = torch.tensor([[2.0]]), torch.tensor([[-1.0]])  # R_n < 0, as on a positive with 1/theta > 1
    obj, Rp, Rn, reg = objective(S, w1, w0, 0.5, U, V, nnpu=True)
    assert Rn < 0 and abs(obj - (-Rn + reg)) < 1e-6
    plain, *_ = objective(S, w1, w0, 0.5, U, V, nnpu=False)
    assert abs(plain - (Rp + Rn + reg)) < 1e-6
    # With R_n >= 0 nnPU is the plain objective.
    w0 = torch.tensor([[1.0]])
    a, *_ = objective(S, w1, w0, 0.5, U, V, nnpu=True)
    b, *_ = objective(S, w1, w0, 0.5, U, V, nnpu=False)
    assert a == b


def test_a_fit_counts_as_finished_only_at_its_last_step():
    import tempfile

    import pandas as pd

    import run_grid

    cfg = dict(dataset="x", seed=42, phase="grid", family="scar", c=0.1, eta=float("nan"), M=float("nan"), lam_rel=1.0, d=100)
    with tempfile.TemporaryDirectory() as t:
        run_grid.TRAJ = Path(t) / "trajectories.csv"
        pd.DataFrame([{**cfg, "step": 10}]).to_csv(run_grid.TRAJ, index=False)
        assert run_grid.key(cfg) not in run_grid.finished()
        pd.DataFrame([{**cfg, "step": run_grid.last_step("scar")}]).to_csv(run_grid.TRAJ, index=False)
        assert run_grid.key(cfg) in run_grid.finished()


def test_nnpu_trajectory_leaves_relmf_only_after_negative_risk():
    from types import SimpleNamespace

    from models import fit_pointwise

    Y = (np.random.default_rng(0).uniform(size=(20, 30)) < 0.2).astype(np.float32)
    u, i = np.nonzero(Y)
    one = np.ones(len(u), np.float32)
    L = make_lists(u, i, one, one, row=np.arange(len(u)))
    d = SimpleNamespace(Y=Y, n_i=Y.sum(0).astype(np.float64), m=20, n=30, fit=np.c_[u, i], rand=np.c_[u, i],
                        val_a=L, val_b=L, test=L, slices={}, rare=np.ones(30, bool))
    cfg = dict(family="relmf", c=None, eta=0.5, M=0.0, lam_rel=0.001, d=5)
    a, _ = fit_pointwise(d, cfg, 0, 600)
    b, _ = fit_pointwise(d, {**cfg, "family": "nnpu"}, 0, 600)
    first = next(t for t, r in enumerate(a) if r["R_n"] < 0)
    assert all(a[t]["J"] == b[t]["J"] for t in range(first)) and a[-1]["J"] != b[-1]["J"]


def test_theta_is_eq_12_and_ap_caps_at_k():
    assert theta(np.array([1.0, 4.0]), 0.5).tolist() == [0.5, 1.0]
    assert ap(np.array([[0, 1, 2]]), np.array([[1.0, 1.0, 1.0]]), 1)[0] == 1


if __name__ == "__main__":
    tests = [f for k, f in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} tests passed")
