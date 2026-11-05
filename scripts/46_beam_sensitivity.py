#!/usr/bin/env python3
"""
Beam sensitivity sweep for the JMGM revision package.

Writes results incrementally after each (A,B,T,ranker) so long runs are resumable.
Default: Effect MLP full grid + Direct ridge at the same grid for residuals.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from itertools import product
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_protocol_freeze():
    path = ROOT / "scripts" / "40_protocol_freeze.py"
    spec = importlib.util.spec_from_file_location("protocol_freeze", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _append_csv(path: Path, row: dict) -> None:
    df = pd.DataFrame([row])
    write_header = not path.exists() or path.stat().st_size == 0
    df.to_csv(path, mode="a", header=write_header, index=False)


def _rebuild_residuals(csv_path: Path, out_path: Path) -> pd.DataFrame:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    wide_rows = []
    for (a, b, t), g in df.groupby(["A", "B", "T"]):
        m = g.drop_duplicates("ranker").set_index("ranker")
        if "effect_mlp" in m.index and "direct_ridge" in m.index:
            wide_rows.append(
                {
                    "A": int(a),
                    "B": int(b),
                    "T": int(t),
                    "mlp": 100.0 * float(m.loc["effect_mlp", "success_rate"]),
                    "direct": 100.0 * float(m.loc["direct_ridge", "success_rate"]),
                    "delta_pp": 100.0
                    * (
                        float(m.loc["effect_mlp", "success_rate"])
                        - float(m.loc["direct_ridge", "success_rate"])
                    ),
                    "mlp_q": float(m.loc["effect_mlp", "mean_queries"]),
                    "direct_q": float(m.loc["direct_ridge", "mean_queries"]),
                }
            )
    wide = pd.DataFrame(wide_rows)
    if len(wide):
        wide = wide.sort_values(["T", "A", "B"])
    wide.to_csv(out_path, index=False)
    return wide


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/beam_sensitivity")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--A", type=int, nargs="+", default=[3, 5, 8])
    parser.add_argument("--B", type=int, nargs="+", default=[3, 5, 8])
    parser.add_argument("--T", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--skip-direct", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Skip (ranker,A,B,T) already present in sensitivity.csv")
    parser.add_argument("--reset", action="store_true",
                        help="Delete prior CSV before running")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    csv_path = args.out / "sensitivity.csv"
    resid_path = args.out / "sensitivity_residuals.csv"
    if args.reset:
        for p in (csv_path, resid_path, args.out / "summary.json"):
            if p.exists():
                p.unlink()

    done = set()
    if args.resume and csv_path.exists() and csv_path.stat().st_size > 0:
        prev = pd.read_csv(csv_path)
        for r in prev.itertuples(index=False):
            done.add((str(r.ranker), int(r.A), int(r.B), int(r.T)))
        print(f"Resuming; {len(done)} configs already done", flush=True)

    pf = _load_protocol_freeze()
    from src.beam_planner import EffectScorer

    store = pf.EditGraphStore(ROOT / "data/processed/search_graph.duckdb")
    episodes = pd.read_parquet(
        ROOT / "data/processed/episode_suites/endpoint_walk.parquet"
    )
    if args.limit:
        episodes = episodes.head(int(args.limit)).reset_index(drop=True)

    ckpt = ROOT / "checkpoints" / "v2" / "effect_predictor.pt"
    if not ckpt.exists():
        ckpt = ROOT / "checkpoints" / "effect_predictor.pt"
    effect = EffectScorer(str(ckpt))

    direct = None
    if not args.skip_direct:
        train = ROOT / "data/processed/effect_splits_v2/train.parquet"
        direct = pf.load_or_train_direct(train, max_rows=15000, seed=42)

    grid = list(product(args.A, args.B, args.T))
    (args.out / "plan.json").write_text(
        json.dumps(
            {
                "n_episodes": len(episodes),
                "checkpoint": str(ckpt),
                "grid": [{"A": a, "B": b, "T": t} for a, b, t in grid],
            },
            indent=2,
        )
    )

    methods = [("effect_mlp", effect, "effect")]
    if direct is not None:
        methods.append(("direct_ridge", direct, "direct"))

    for a, b, t in grid:
        for name, scorer, ranker in methods:
            key = (name, a, b, t)
            if key in done:
                print(f"skip {key}", flush=True)
                continue
            succ = []
            queries = []
            t0 = time.perf_counter()
            for _, ep in episodes.iterrows():
                ep_d = ep.to_dict()
                target = pf.Properties(
                    ep_d["target_qed"], ep_d["target_logp"], ep_d["target_mw"]
                )
                planner = pf.BeamPlanner(
                    store=store,
                    scorer=scorer,
                    scales=pf.SCALES,
                    tolerances=pf.TOL,
                    beam_width=int(b),
                    actions_per_state=int(a),
                    max_outgoing=500,
                    max_steps=int(t),
                    replace_only=True,
                    ranker=ranker,
                    oracle_correction=True,
                    seed=42 + int(ep_d["episode"]),
                )
                planner.dedupe_queries = True
                planner.count_start_query = False
                result = planner.plan(target, start_smiles=ep_d["start_smiles"])
                err = np.abs(result.properties.array() - target.array())
                ok = bool(np.all(err <= np.asarray(pf.TOL)))
                succ.append(ok)
                queries.append(int(planner.last_stats["oracle_calls"]))
            row = {
                "ranker": name,
                "A": a,
                "B": b,
                "T": t,
                "success_rate": float(np.mean(succ)),
                "mean_queries": float(np.mean(queries)),
                "n": len(succ),
                "seconds": time.perf_counter() - t0,
            }
            _append_csv(csv_path, row)
            print(row, flush=True)
            wide = _rebuild_residuals(csv_path, resid_path)
            if len(wide):
                print(wide.to_string(index=False), flush=True)

    wide = _rebuild_residuals(csv_path, resid_path)
    summary = {
        "n_episodes": int(len(episodes)),
        "n_rows": int(pd.read_csv(csv_path).shape[0]) if csv_path.exists() else 0,
        "residuals": wide.to_dict(orient="records") if len(wide) else [],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
