# EditGraph reproducibility checklist

This archive contains code, environment pins, frozen manifests (SHA-256), and
scripts to regenerate headline tables for the EditGraph label-access audit.

## Large artifacts (not always bundled)
Obtain under the hashes in `results/canonical_frozen200/manifest.json`:
- `data/processed/search_graph.duckdb`
- `data/processed/episode_suites/endpoint_walk.parquet`
- `data/processed/effect_splits_v2/train.parquet`

Place them at those relative paths, then:

```bash
conda env create -f environment.yml
conda activate causalmol
python scripts/44_consistency_check_main_tables.py
python scripts/40_protocol_freeze.py --skip-partial
python scripts/41_exact_baselines_unique_node.py
python scripts/43_frozen_stratified_diagnostics.py
```
