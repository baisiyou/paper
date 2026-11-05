# EditGraph reproducibility package

Code, environment pins, frozen manifests (SHA-256), checkpoint, and scripts
supporting the manuscript:

**Auditing Multi-Property Navigation and Label Access on Molecular Edit Graphs**

Authors: Yuran Zhang (McGill University); Rubing Zhang (Vanier College).

This repository contains only what is needed to regenerate and verify the
principal tables (and the revision audit artifacts). Manuscript PDFs and
editorial response letters are submitted separately via the journal system.

## Quick start

```bash
conda env create -f environment.yml
conda activate causalmol
python scripts/44_consistency_check_main_tables.py
```

Primary regeneration scripts:

```bash
python scripts/40_protocol_freeze.py --skip-partial
python scripts/41_exact_baselines_unique_node.py
python scripts/43_frozen_stratified_diagnostics.py
```

Additional revision audits (optional):

```bash
python scripts/47_split_leakage_audit.py
python scripts/46_beam_sensitivity.py   # requires local DuckDB index + checkpoint
```

See `docs/REVIEW_REPRODUCIBILITY.md` and `README_REVIEW.md` for the full checklist.

**Versioned revision tag:** `jmgm-d-26-02110-r1`
(https://github.com/baisiyou/paper/releases/tag/jmgm-d-26-02110-r1).

## Large artifacts (not in git)

Place these under the SHA-256 hashes recorded in
`results/canonical_frozen200/manifest.json`:

- `data/processed/search_graph.duckdb`
- `data/processed/episode_suites/endpoint_walk.parquet`
- `data/processed/effect_splits_v2/train.parquet`

Raw MOSES molecules follow the original MOSES distribution terms.

## Contents

| Path | Role |
|------|------|
| `src/` | Search, effect model, baselines |
| `scripts/` | Protocol freeze, exact baselines, diagnostics, revision audits |
| `checkpoints/v2/` | Effect MLP weights + metrics |
| `results/` | Frozen manifests and table summaries |
| `configs/` | Model config |
