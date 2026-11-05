"""Paths, seeds and the experiment grid. Every other module reads its constants from here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC, RESULTS = ROOT / "data" / "raw", ROOT / "data" / "processed", ROOT / "results"
TRAJ = PROC / "trajectories.csv"

SPLIT_SEED, USER_SPLIT_SEED, BOOTSTRAP_SEED = 12345, 0, 42
MODEL_SEEDS = (42, 43, 44)
TORCH_THREADS = 1  # 4 threads are only about 1.25x faster per step

THRESHOLD = 4       # rating >= 4 counts as a positive
VAL_NEGATIVES = 100  # unlabelled candidates per validation user
SNIPS_FLOOR = 0.01  # theta' = max(theta(eta=0.5), 0.01), for every family
DIM, LR, EVERY = 100, 0.005, 10
# Step budgets. Doubled from (naive 3000, others 1500) for families whose validation winner sat in the last 5%
# of its budget. Rel-MF follows nnPU so that their last-step comparison is at the same step.
STEPS = {"naive": 6000, "scar": 3000, "nnpu": 3000, "balanced": 3000, "relmf": 3000, "relmf_eta025": 3000}
DEFAULT_STEPS = 1500
LAM_REL = (0.3, 1, 3, 10)  # lambda = lam_rel * mean positive weight
SENS_DIMS = (30, 200)
BPR = dict(lr=0.01, batch=4096, epochs=100, regs=(1e-3, 1e-2, 1e-1, 1.0))
EASE_LAMS = (10, 30, 100, 300, 1000, 3000, 10000)

# Model selection rule -> trajectory column it maximises.
RULES = {
    "A0": "val_a_naive", "A": "val_a_snips", "A+": "val_a_pooled", "B": "val_b",
    "B5a": "B5a", "B5b": "B5b", "B2.5a": "B2.5a", "B2.5b": "B2.5b", "B2.5c": "B2.5c", "B2.5d": "B2.5d",
    "oracle": "test_dcg5",
}
FINAL_RULES = {"relmf_eta025": ("A",), "relmf_eta1": ("A",)}  # everything else: A and B
DETERMINISTIC = ("popularity", "ease")  # one fit is enough, whatever the seed

NAN = float("nan")


def _cfg(family, lam_rel=NAN, c=NAN, eta=NAN, M=NAN, d=DIM):
    return dict(family=family, c=c, eta=eta, M=M, lam_rel=lam_rel, d=d)


def _family(family, lam):
    """Every configuration of one pointwise family at one lam_rel."""
    if family == "naive":
        return [_cfg(family, lam)]
    if family == "wmf":
        return [_cfg(family, lam, c=c) for c in (3, 10, 30, 100, 300, 1000)]  # the paper stops at 100; 300 and 1000 because the upper limit won
    if family == "scar":
        return [_cfg(family, lam, c=1 / c) for c in (3, 10, 30, 100)]
    if family == "relmf":
        return [_cfg(family, lam, eta=0.5, M=M) for M in (0, 0.1, 0.3)]
    if family == "relmf_eta025":
        return [_cfg(family, lam, eta=0.25, M=0)]
    if family == "relmf_eta1":
        return [_cfg(family, lam, eta=1, M=M) for M in (0, 0.03, 0.1)]
    return [_cfg(family, lam, eta=0.5, M=0)]  # nnpu, balanced


def last_step(family):
    """The last logged step of a fit: 0 for the closed-form models, epochs for BPR-MF."""
    return 0 if family in DETERMINISTIC else BPR["epochs"] if family == "bpr" else STEPS.get(family, DEFAULT_STEPS)


POINTWISE = ("naive", "wmf", "scar", "relmf", "relmf_eta025", "relmf_eta1", "nnpu", "balanced")
# Extra lam_rel for main-table families whose winner sat on the edge of LAM_REL under a validation rule.
EXTRA_LAM = {"coat": {"naive": (0.1, 0.03), "wmf": (0.1,), "balanced": (0.1, 0.03, 0.01)},
             "yahoo": {"relmf": (0.1,), "nnpu": (0.1,), "balanced": (0.1,)}}


def control(dataset):
    """SCAR at balanced's scale of positive weights on Yahoo (mean about 500): the nearest cell with the PU term and no
    popularity shape. Apart from the SCAR grid, so the SCAR winner does not move. Coat's SCAR grid comes close to balanced's scale there (c′ = 100 against about 129)."""
    return [_cfg("scar", lam, c=1 / 500) for lam in (0.1,) + LAM_REL] if dataset == "yahoo" else []


def grid(dataset):
    """80 pointwise fits plus edge extensions, BPR, EASE, popularity. lam_rel holds BPR's reg and EASE's lambda."""
    out = [c for lam in LAM_REL for f in POINTWISE for c in _family(f, lam)]
    out += [c for f, lams in EXTRA_LAM[dataset].items() for lam in lams for c in _family(f, lam)]
    out += [_cfg("bpr", reg) for reg in BPR["regs"]]
    out += [_cfg("ease", lam) for lam in EASE_LAMS]
    out.append(_cfg("popularity"))
    return out


FAMILIES = ("popularity", "naive", "bpr", "ease", "wmf", "scar", "relmf", "nnpu", "balanced", "relmf_eta025", "relmf_eta1")
LABELS = {
    "popularity": "Popularity", "naive": "Naive MF", "bpr": "BPR-MF", "ease": "EASE",
    "wmf": "WMF", "scar": "SCAR (constant θ)", "relmf": "Rel-MF", "nnpu": "Rel-MF + nnPU",
    "balanced": "Rel-MF balanced (official-code objective)",
    "relmf_eta025": "Rel-MF, η = 0.25", "relmf_eta1": "Rel-MF, η = 1", "scar_c500": "SCAR, c′ = 500",
}

# Figures: the first three categorical slots of a colour-blind-checked palette, plus recessive neutrals.
BLUE, ORANGE, AQUA, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#c9c8c1", "#0b0b0b", "#52514e"
MPL = {
    "figure.dpi": 110, "savefig.dpi": 180, "savefig.bbox": "tight", "font.size": 10, "lines.linewidth": 2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GRAY, "axes.grid": True,
    "axes.axisbelow": True, "grid.color": "#ecebe6", "grid.linewidth": 0.8, "axes.labelcolor": INK2,
    "axes.titlesize": 10, "axes.titlelocation": "left", "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "legend.frameon": False,
}
