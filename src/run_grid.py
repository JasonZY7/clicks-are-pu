"""Grid, finals and d sensitivity. Appends to data/processed; finished fits are skipped on restart.

python src/run_grid.py                                    everything, in order, on one process
python src/run_grid.py --dataset yahoo --phase grid --shard 0/3   one of three parallel workers
"""
import argparse
import fcntl
import os
import sys
import time

import pandas as pd

from config import DEFAULT_STEPS, DETERMINISTIC, FAMILIES, FINAL_RULES, MODEL_SEEDS, PROC, SENS_DIMS, STEPS, TRAJ, control, grid, last_step
from data import load
from evaluate import load_traj, select
from models import fit

KEY = ["dataset", "family", "c", "eta", "M", "lam_rel", "d", "seed", "phase"]


def key(row):
    return "|".join(row[k] if isinstance(row[k], str) else f"{row[k]:.6g}" for k in KEY)


def job_key(dataset, job):
    return key(dict(dataset=dataset, seed=job[1], phase=job[2], **job[0]))


def finished():
    """Fits whose rows reach their family's last step, so a raised budget or a fit cut at a row boundary runs the fit again.

    A row cut part way breaks load_traj: cut the file back to its last newline before restarting.
    """
    if not TRAJ.exists():
        return set()
    df = pd.read_csv(TRAJ, usecols=KEY + ["step"]).groupby(KEY, dropna=False, sort=False).step.max().reset_index()
    return {key(r) for r in df.to_dict("records") if r["step"] >= last_step(r["family"])}


def append(df):
    """A single locked append, so parallel workers cannot interleave rows."""
    PROC.mkdir(parents=True, exist_ok=True)
    with open(TRAJ, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(df.to_csv(header=os.fstat(f.fileno()).st_size == 0, index=False))


def steps_of(family):
    return STEPS.get(family, DEFAULT_STEPS)


def parts_dir(dataset):
    return PROC / "final_parts" / dataset


def pending(dataset, job, done):
    """Not finished, or finished earlier for other rules: its scores file lacks a rule this job must keep."""
    cfg, seed, phase, rules = job
    if job_key(dataset, job) not in done:
        return True
    return bool(rules) and not set(rules) <= set(pd.read_parquet(part_path(dataset, cfg, seed, phase), columns=["rule"]).rule)


def run(d, jobs, shard):
    """jobs = (cfg, seed, phase, rules). Scores are written before the trajectory row that marks a fit as done."""
    done = finished()
    i, n = shard  # split before filtering, so the partition does not depend on when each worker read the file
    jobs = [j for j in jobs[i::n] if pending(d.name, j, done)]
    for pos, (cfg, seed, phase, rules) in enumerate(jobs):
        t0 = time.time()
        rows, scores = fit(d, cfg, seed, steps_of(cfg["family"]), rules)
        if scores:
            parts_dir(d.name).mkdir(parents=True, exist_ok=True)
            family = {"calib": "scar_c10", "control": "scar_c500"}.get(phase, cfg["family"])
            df = pd.concat([pd.DataFrame(dict(family=family, rule=r, seed=seed, row_id=range(len(s)), score=s))
                            for r, s in scores.items()])
            df.to_parquet(part_path(d.name, cfg, seed, phase), index=False)
        append(pd.DataFrame(rows).assign(dataset=d.name, seed=seed, phase=phase, **cfg))
        best = max(rows, key=lambda r: r["val_b"])
        print(f"[{d.name}] {phase} {pos + 1}/{len(jobs)} {cfg['family']} c={cfg['c']:.3g} eta={cfg['eta']:.3g} M={cfg['M']:.3g} "
              f"lam={cfg['lam_rel']:.3g} d={cfg['d']} seed={seed} {time.time() - t0:.0f}s | best val_b step {best['step']} "
              f"test {best['test_dcg5']:.4f} rare {best['rare_credit5']:.4f} | last test {rows[-1]['test_dcg5']:.4f} R_n {rows[-1]['R_n']:+.5f}",
              flush=True)


def cfg_of(row):
    return dict(family=row.family, c=float(row.c), eta=float(row.eta), M=float(row.M), lam_rel=float(row.lam_rel), d=int(row.d))


def report(traj):
    """Each family's winners. Warns when a winner sits on the edge of its step budget or of a grid."""
    for family in FAMILIES:
        rows = traj[(traj.family == family) & (traj.phase == "grid")]
        for rule in ("A", "B"):
            r = select(traj, family, rule)
            msg = f"{family:13s} {rule} c={r.c:.3g} eta={r.eta:.3g} M={r.M:.3g} lam={r.lam_rel:.3g} step={r.step:.0f} " \
                  f"test={r.test_dcg5:.4f} rare={r.rare_credit5:.4f}"
            if family not in DETERMINISTIC + ("bpr",) and r.step >= 0.95 * steps_of(family):
                msg += "  WARNING: best step in the last 5%"
            if rows.lam_rel.nunique() > 1 and r.lam_rel in (rows.lam_rel.min(), rows.lam_rel.max()):
                msg += "  WARNING: lambda on the grid edge"
            if rows.c.nunique() > 1 and r.c in (rows.c.min(), rows.c.max()):
                msg += "  WARNING: c on the grid edge"
            print(f"[{traj.dataset.iloc[0]}] {msg}", flush=True)


def winners(traj, family, rules, phase="grid"):
    """[(cfg, the rules it wins)], one entry per distinct configuration."""
    by_cfg = {}
    for rule in rules:
        cfg = cfg_of(select(traj, family, rule, phase))
        same = "|".join(f"{v}" for v in cfg.values())  # NaNs do not compare equal, their text does
        by_cfg.setdefault(same, (cfg, []))[1].append(rule)
    return list(by_cfg.values())


def final_jobs(traj):
    jobs = []
    for family in FAMILIES:
        seeds = MODEL_SEEDS[:1] if family in DETERMINISTIC else MODEL_SEEDS
        for cfg, rules in winners(traj, family, FINAL_RULES.get(family, ("A", "B"))):
            jobs += [(cfg, seed, "final", tuple(rules)) for seed in seeds]
    # Calibration check in NB2: SCAR at c = 1/10, close to the label frequency c*, one seed.
    scar10 = traj[(traj.family == "scar") & (traj.c.round(6) == 0.1)]
    jobs.append((cfg_of(select(scar10, "scar", "A")), MODEL_SEEDS[0], "calib", ("A",)))
    if (traj.phase == "control_grid").any():
        for cfg, rules in winners(traj, "scar", ("A", "B"), "control_grid"):
            jobs += [(cfg, seed, "control", tuple(rules)) for seed in MODEL_SEEDS]
    return jobs


def sens_jobs(traj):
    return [(cfg_of(select(traj, f, "A")) | dict(d=dim), MODEL_SEEDS[0], "sens", ()) for f in ("naive", "relmf") for dim in SENS_DIMS]


def part_path(dataset, cfg, seed, phase):
    return parts_dir(dataset) / f"{key(dict(dataset=dataset, seed=seed, phase=phase, **cfg)).replace('|', '_')}.parquet"


def merge_scores(dataset, jobs):
    """The current final fits of a dataset, in one parquet file: (family, rule, seed, row_id, score).

    Returns without writing while any fit is pending, so an older file stays.
    A row is appended after its scores are written, so a part with a row is complete and current.
    """
    done = finished()
    missing = [j for j in jobs if pending(dataset, j, done)]
    if missing:
        print(f"[{dataset}] {len(missing)} of {len(jobs)} final fits missing, final_scores not written", flush=True)
        return
    parts = []
    for cfg, seed, phase, rules in jobs:
        part = pd.read_parquet(part_path(dataset, cfg, seed, phase))
        parts.append(part[part.rule.isin(rules + tuple(f"last_{r}" for r in rules))])
    df = pd.concat(parts)
    assert not df.duplicated(["family", "rule", "seed", "row_id"]).any()
    tmp = PROC / f"final_scores_{dataset}.{os.getpid()}.tmp"
    df.to_parquet(tmp, index=False)
    tmp.replace(PROC / f"final_scores_{dataset}.parquet")


def require_grid(name):
    """Selection on a partial grid picks the wrong winners without any error."""
    done = finished()
    missing = [c for c in grid(name) if key(dict(dataset=name, seed=MODEL_SEEDS[0], phase="grid", **c)) not in done]
    missing += [c for c in control(name) if key(dict(dataset=name, seed=MODEL_SEEDS[0], phase="control_grid", **c)) not in done]
    if missing:
        sys.exit(f"[{name}] {len(missing)} grid fits missing, finish --phase grid first")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["coat", "yahoo"], action="append")
    ap.add_argument("--phase", choices=["all", "grid", "report", "finals", "sens"], default="all")
    ap.add_argument("--shard", default="0/1", help="i/n: run every n-th job, starting at i")
    args = ap.parse_args()
    shard = tuple(map(int, args.shard.split("/")))
    for name in args.dataset or ["coat", "yahoo"]:
        d = load(name)
        if args.phase in ("all", "grid"):
            run(d, [(c, MODEL_SEEDS[0], "grid", ()) for c in grid(name)] + [(c, MODEL_SEEDS[0], "control_grid", ()) for c in control(name)], shard)
        if args.phase == "grid":
            continue
        if args.phase != "report":
            require_grid(name)
        traj = load_traj(name)
        if args.phase in ("all", "report"):
            report(traj)
        if args.phase in ("all", "finals"):
            jobs = final_jobs(traj)
            run(d, jobs, shard)
            merge_scores(name, jobs)
        if args.phase in ("all", "sens"):
            run(d, sens_jobs(traj), shard)


if __name__ == "__main__":
    main()
