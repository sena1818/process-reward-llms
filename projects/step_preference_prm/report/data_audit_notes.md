# Data Audit Notes

The first project question is empirical:

> Does PRM800K contain enough same-prefix candidate alternatives with different
> labels to support step-level preference modeling?

Run the minimum workflow:

```bash
python scripts/download_prm800k.py --all
python scripts/inspect_prm800k.py --config configs/audit.yaml
python scripts/00_audit_data.py --config configs/audit.yaml
```

The useful outputs are:

- `outputs/audit/sample_preview.md`
- `outputs/audit/sample_records.json`
- `outputs/audit/data_audit.json`
- `outputs/audit/pair_stats.json`

Decision rule for the project idea:

- If exact-prefix preference pairs are plentiful, make pairwise/hybrid PRM the
  main method.
- If exact-prefix preference pairs are sparse, keep pointwise ordinal PRM as
  the main baseline and use pairwise loss as an auxiliary signal.
