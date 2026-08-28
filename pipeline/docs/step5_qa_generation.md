# Step 5 — QA Generation

Turns Step-2/3 outputs into structured indexes and then into deterministic QA,
using the Step-4 templates. All outputs land under `config.json` paths
(`index_dir`, `qa_out`). Every stage is resume-safe at shard level.

Order:

```bash
python step5_build_index.py                 # 5a  pool_meta + sim meta -> full_index.pkl, meta_all.csv
python step5_region_disp.py         # 5b  per-mesh regional (dx,dz,|d|) vs rest -> region_disp.npz
python step5_extract_features.py            # 5c  32 palate-normalized features -> features_all.npy (+ok/keys)

python step5_gen_physics_qa.py --lang ko 0 295157     # 5d  physics chain per VALID mesh
python step5_gen_physics_qa.py --lang en 0 295157
python step5_gen_feature_qa.py --lang ko A1 0 295157  # 5e  shape description per VALID+feat_ok mesh
python step5_gen_feature_qa.py --lang ko B3           #     dose-response sweeps (single + effort)
python step5_gen_feature_qa.py --lang en A1 0 295157
python step5_gen_feature_qa.py --lang en B3
```

Data dependencies:

| stage | needs |
|---|---|
| build_index | `pool_meta.csv`, `{sim_out}/meta/shard_*.csv` |
| compute_region_disp | `{sim_out}/verts/*`, rest mesh (`config rest_index`) |
| extract_features | `{sim_out}/verts/*`, `assets/model_palate.csv` |
| gen_physics_qa | region_disp, pool_meta, sim meta, `settings/centers.csv` |
| gen_feature_qa A1 | features_all + full_index |
| gen_feature_qa B3 | region_disp + full_index |

Only 5b/5c touch the mesh binaries; QA generation itself runs from the compact
indexes (fast, re-runnable — new question families can be added without any
re-simulation).

`scripts/_bootstrap.py` only inserts the pipeline root on `sys.path`; run the scripts
from this directory.
