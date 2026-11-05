#!/usr/bin/env python3
"""
Split-leakage audit for the JMGM revision package.

Quantifies, across train/val/test endpoint splits:
  - Bemis-Murcko scaffold Jaccard
  - BRICS fragment set overlap
  - nearest-neighbor Morgan Tanimoto (test→train)

Output: results/split_leakage_audit/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def scaffolds(smiles: list[str]) -> set[str]:
    out = set()
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        try:
            out.add(MurckoScaffold.MurckoScaffoldSmiles(mol=mol))
        except Exception:
            continue
    return out


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    return float(len(a & b) / len(u)) if u else 0.0


def morgan_fps(smiles: list[str], n_bits: int = 2048, radius: int = 2):
    fps = []
    keep = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits))
        keep.append(s)
    return keep, fps


def nn_tanimoto(query_fps, ref_fps) -> np.ndarray:
    if not query_fps or not ref_fps:
        return np.array([])
    vals = []
    for q in query_fps:
        sims = DataStructs.BulkTanimotoSimilarity(q, ref_fps)
        vals.append(max(sims) if sims else 0.0)
    return np.asarray(vals, dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits",
        type=Path,
        default=ROOT / "data/processed/effect_splits_v2",
    )
    parser.add_argument(
        "--annotated",
        type=Path,
        default=ROOT / "data/processed/moses_annotated.parquet",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results/split_leakage_audit",
    )
    parser.add_argument("--max-endpoints", type=int, default=5000,
                        help="Cap unique endpoints per split for Tanimoto NN")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    splits = {}
    for name in ("train", "val", "test"):
        path = args.splits / f"{name}.parquet"
        if not path.exists():
            print(f"[error] missing {path}")
            return 1
        df = pd.read_parquet(path)
        ends = sorted(set(df["smiles_a"].astype(str)) | set(df["smiles_b"].astype(str)))
        if len(ends) > args.max_endpoints:
            ends = list(rng.choice(ends, size=args.max_endpoints, replace=False))
        splits[name] = ends

    # Fragment sets from annotated table if available
    frag_map: dict[str, set[str]] = {}
    if args.annotated.exists():
        ann = pd.read_parquet(args.annotated, columns=["smiles", "brics_fragments"])
        for s, fr in zip(ann["smiles"], ann["brics_fragments"]):
            if isinstance(fr, (list, tuple, np.ndarray)):
                frag_map[str(s)] = set(str(x) for x in fr if x)
            else:
                frag_map[str(s)] = set()

    report: dict = {"splits": {k: len(v) for k, v in splits.items()}, "pairs": {}}

    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        sa, sb = set(splits[a]), set(splits[b])
        sc_a, sc_b = scaffolds(list(sa)), scaffolds(list(sb))
        entry = {
            "endpoint_jaccard": jaccard(sa, sb),
            "scaffold_jaccard": jaccard(sc_a, sc_b),
            "n_scaffold_a": len(sc_a),
            "n_scaffold_b": len(sc_b),
        }
        if frag_map:
            fa = set().union(*(frag_map.get(s, set()) for s in sa))
            fb = set().union(*(frag_map.get(s, set()) for s in sb))
            entry["fragment_jaccard"] = jaccard(fa, fb)
            entry["n_fragments_a"] = len(fa)
            entry["n_fragments_b"] = len(fb)
            entry["fragment_intersection"] = len(fa & fb)

        # NN Tanimoto: b → a
        keep_a, fps_a = morgan_fps(list(sa))
        keep_b, fps_b = morgan_fps(list(sb))
        nn = nn_tanimoto(fps_b, fps_a)
        if len(nn):
            entry["nn_tanimoto_mean"] = float(nn.mean())
            entry["nn_tanimoto_median"] = float(np.median(nn))
            entry["nn_tanimoto_p90"] = float(np.quantile(nn, 0.9))
            entry["frac_nn_ge_0.7"] = float((nn >= 0.7).mean())
            entry["frac_nn_ge_0.5"] = float((nn >= 0.5).mean())
        report["pairs"][f"{a}_vs_{b}"] = entry
        print(f"{a} vs {b}: {entry}")

    (args.out / "split_leakage.json").write_text(json.dumps(report, indent=2))
    # Flat CSV for SI table
    rows = []
    for pair, entry in report["pairs"].items():
        rows.append({"pair": pair, **entry})
    pd.DataFrame(rows).to_csv(args.out / "split_leakage.csv", index=False)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
