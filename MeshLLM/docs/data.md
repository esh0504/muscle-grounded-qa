# Data preparation

Everything lives under `DATA/` (in Docker this is the `${DATA_DIR}` volume;
bare metal it can be a directory or symlink at the repo root).

## Expected layout

```
DATA/mesh/
    topology.obj                       fixed surface topology (370 verts)
    verts/shard_%05d.bin               big-endian float32 (n, 370, 3), 1000/shard
    meta_all.csv                       per-index validity labels (VALID / ...)
    pool_meta.csv                      per-index muscle activations + section/phoneme meta
    train.txt  val.txt  test.txt       index splits (generated, see below)
DATA/qa/{en,ko}/nat_out/nat_*.jsonl    naturalized QA (multiturn + mask_spans)
DATA/unseentest/                       unseen eval sets (built by tools/build_set1_probe.py)
DATA/aux/                              probe-builder inputs (centers.csv, region_disp.npz, properties.jsonl)
```

## From the 3DTongueQA pipeline outputs

**Easiest: point `DATA_DIR` at the pipeline's `DATA_DIR` directly.** The
pipeline writes `mesh/` (pool + topology + verts, and `meta_all.csv` after its
`qa` service ran once), `index/` and `qa/` — which is already this layout, so
the scripts use it as-is (they link `mesh/meta_all.csv` from `index/` when
needed, and generate the splits into `mesh/`). Stage 2 falls back to the
pipeline's template QA under `qa/` when no naturalized corpus is present.

(Older pipeline outputs with `static_300k/` + `outputs/` are also detected —
the scripts build the `mesh/` view with relative symlinks, no copying.)

Then generate the stratified split (9 : 0.5 : 0.5 per section, shared by
Stage 1 and Stage 2 so their val/test mesh boundaries agree):

```bash
python datasets/split_trainvaltest.py --data DATA/mesh --out DATA/mesh
```

(`scripts/train_s1.sh` / `scripts/train_s2.sh` run this automatically when
`train.txt`/`val.txt` are missing.)

## Naturalized vs template QA

Training uses the naturalized corpus (`nat_*.jsonl`: LLM-rewritten surface
form, gold facts fixed, `mask_spans` recomputed and verified). The
naturalization step is part of the research pipeline, not this release. To
train directly on the pipeline's template QA instead, point the dataset at the
raw files, e.g.

```bash
python train_s2.py +experiment=ours_en 'datasets.qa_glob=DATA/qa/physics_en/qa_en_*.jsonl'
```

(the record schema — `mesh_ref`, `conversations`, `mask_spans` — is identical).

## Unseen evaluation sets

`tools/build_set1_probe.py` builds the Set-1 structured probe from held-out
test meshes; the Set-1/2/3 generation configs are under `configs/eval*` and
`configs/evaluators/`.
