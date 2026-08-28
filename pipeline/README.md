# 3DTongueQA Construction Pipeline

Refactored, release-ready implementation of the data-construction framework in
*"A Simulator-Grounded Framework for Constructing Verifiable Muscle-Grounded QA
from 3D Tongue Meshes"* (ICASSP submission). Controlled 11-D muscle activations
are mapped through the ArtiSynth Badin FEM tongue model to screened 3D meshes,
converted into structured records, and rendered as deterministic QA (KO/EN).

```
settings/   everything the user configures (paths, anchors.yaml, sampling.yaml, centers.csv)
modules/    ALL Python code:
              config / muscles / anchors / sampling_design / dorsum / features /
              mesh_io / spans                        shared library
              qa_templates/                          Step 4 — KO/EN surface forms + span tagging
              steps/                                 step entry modules (run via python -m)
              artisynth/                             Step 3 — Jython exporter + Windows .bat
scripts/    TWO shell entry points:
              make_mesh.sh [START END NPROC]         steps 1-3: spec check -> pool -> FEM meshes
              make_qa.sh                             steps 4-5: indexes -> physics/feature QA (KO+EN)
assets/     fixed data (model_palate.csv)
docs/       per-step design docs + docker guide
plus at the root: Dockerfile, docker-entrypoint.sh, docker-compose(.dev).yml, .env
```

See `settings/README.md` for what to configure and `docs/` for per-step details.

## Requirements

- Steps 1, 2, 5: Python 3.9+ with `numpy`, `pandas`, `pyyaml` (any OS).
- Step 3: a compiled [ArtiSynth](https://www.artisynth.org) `artisynth_core`
  checkout with `artisynth_models` (provides
  `artisynth.models.tongue3d.StableFemMuscleTongueDemo`), Java 8+, Windows batch
  runner. The Jython scripts are Python 2 syntax by necessity (ArtiSynth embeds
  Jython) — do not "modernize" them.

## Reproduction (Docker, recommended)

A self-contained release image (ArtiSynth + Python + this code) — see
`docs/docker.md`:

```bash
cp settings/docker.env.example .env      # set DATA_DIR (big disk)
docker compose build
docker compose run --rm mesh             # steps 1-3   (pilot: run --rm mesh 0 1000 1)
docker compose run --rm qa               # steps 4-5
```

## Reproduction (bare metal)

```bash
# 0. edit settings/config.json (paths) + settings/sampling.yaml (total/seed)

# Linux/macOS (ArtiSynth on the same machine):
scripts/make_mesh.sh                                   # steps 1-3 -> {sim_out}/verts|nodes|meta
scripts/make_qa.sh                                     # steps 4-5 -> {index_dir}, {qa_out}

# Windows: steps 1-2 via python, step 3 via the .bat, steps 4-5 via python -m
python -m modules.steps.check_design
python -m modules.steps.sample_pool
modules\artisynth\run_headless.bat 0 295157 4
python -m modules.steps.build_index && python -m modules.steps.region_disp
python -m modules.steps.extract_features --workers 2
python -m modules.steps.gen_physics_qa --lang ko 0 295157     # and --lang en
python -m modules.steps.gen_feature_qa --lang ko A1 0 295157  # and B3, en
```

Every generation stage is shard-resumable: re-running skips finished shard files.

## Record schema

One JSON line per record:

```json
{"index": 123, "mesh_ref": {"verts_shard": 0, "row_in_shard": 123},
 "scenario": "physics_chain | shape_desc | dose_response",
 "lang": "ko | en",
 "conversations": [{"from": "human", "value": "..."},
                   {"from": "gpt", "value": "...", "mask_spans": [...]}],
 "turn_types": ["A2", "D1", "C1", "..."], "verified": true}
```

`mask_spans` tags every muscle name, number (with role), movement and region
token in each gold answer — the machinery behind the record-based faithfulness
check and the span-level metrics (Muscle EM / Value Acc / Direction EM).

## Provenance & verification of this refactor

- `step2` was checked against the original `gen_pool_sections.py`: identical
  `pool` body and metadata, byte for byte, at matched (total, seed).
- `step5` generators were checked against the original `scale_qa[_en].py` and
  `feature_qa[_en].py` on a synthetic end-to-end fixture: all KO/EN physics and
  A1/B3 outputs byte-identical.
- `step3` is the original exporter with the hardcoded machine paths replaced by
  `-Dstatic.*` system properties; the simulation logic is untouched.

## Not included here

Language naturalization (LLM surface rewriting + regeneration loop), the
validation/figure scripts (coverage, F1/F2, precision-recall), and model
training/evaluation live in the research repo; this package is the deterministic
construction pipeline only.
