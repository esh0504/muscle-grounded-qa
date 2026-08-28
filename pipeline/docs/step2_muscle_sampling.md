# Step 2 — Muscle-Activation Sampling

Samples the 11-D activation pool from the Step-1 design.

```bash
python scripts/step2_sample_pool.py                    # total/seed from settings/sampling.yaml (300000 / 0)
python scripts/step2_sample_pool.py --total 300000 --seed 0
```

Outputs (paths from `config.json`):

- `pool.txt` — `index,GGP,...,SL` rows; the direct input of Step 3.
- `pool_meta.csv` — per-index section, phoneme, detail (e.g. `GGP@0.45`,
  `anchor:i_front`, `effort:GH+GGA@0.66`), `base_index`/`delta` (neighbor
  pairs), `n_active`, literature refs, and the activation vector.

Properties guaranteed by construction:

- deterministic given (total, seed) — the RNG call order is preserved from the
  original `gen_pool_sections.py`, so the released 295,157-row pool is exactly
  reproduced with the defaults;
- enumerated primitives (rest, 66 singles, 495 pairs, 3000 triples) always
  present; duplicates removed at 4 decimals; final order shuffled (seed 42)
  with `base_index` remapped, so **any index prefix is a balanced subsample**;
- the all-zero REST sample is pinned at index 0 (`rest_first: true` in
  settings/sampling.yaml), so even a small pilot range contains the
  displacement reference; the released pool used `rest_first: false`
  (rest at 290609) — set that to reproduce it exactly;
- per-muscle cap `AMAX=0.9` and total-activation budgets applied everywhere.
