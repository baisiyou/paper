# EditGraph reproducibility snapshot

Public reproducibility package:
https://github.com/baisiyou/paper
(current main branch of this repository).

## Must include
- `environment.yml`, `requirements.txt`
- `scripts/` (at least `40_protocol_freeze.py`, `41_exact_baselines_unique_node.py`,
  `42_matched_architecture_ablation.py`, `43_frozen_stratified_diagnostics.py`,
  `37_confirm_v2_disjoint.py`, `44_consistency_check_main_tables.py`)
- `src/`
- `results/canonical_frozen200/manifest.json` (suite/checkpoint/DuckDB SHA-256)
- `results/protocol_freeze/f0_identity_check.json`
- `results/protocol_freeze/primary_summary.csv`
- `results/exact_baselines_unique_node/summary.csv`
- `results/benchmark_diagnostics/frozen_stratified_success.json`
- `checkpoints/v2/effect_predictor.pt` (+ metrics JSON) **or** download URL with matching hash
- Pointer/README for obtaining `data/processed/search_graph.duckdb` and episode parquet
  under the hashes in the manifest (large files may be Zenodo-only)

## Optional revision audits
- `scripts/46_beam_sensitivity.py` + `results/beam_sensitivity/`
- `scripts/47_split_leakage_audit.py` + `results/split_leakage_audit/`

## Verify locally
```bash
conda env create -f environment.yml
python scripts/40_protocol_freeze.py --skip-partial
python scripts/41_exact_baselines_unique_node.py
python scripts/43_frozen_stratified_diagnostics.py
python scripts/44_consistency_check_main_tables.py
```
