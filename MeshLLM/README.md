# MeshLLM — 3DTongueQA Baseline Training & Evaluation

Release version of the baseline models used to validate the **3DTongueQA**
dataset (*"A Simulator-Grounded Framework for Constructing Verifiable
Muscle-Grounded QA from 3D Tongue Meshes"*, ICASSP submission). Two stages:

```
Stage 1   mesh displacement -> 11 muscle activations      SpiralNet++ regression
              |  (the muscle-aware pretraining of the paper)
              v  encoder FROZEN and reused
Stage 2   (mesh, question) -> answer                      mesh encoder -> bridge
                                                          -> Qwen3-8B (LoRA)
```

The core claim tested here: *a 3D encoder pretrained with muscle regression can
replace the vision encoder of a LLaVA-style multimodal LLM*. One experiment
overlay per row in `configs/experiment/`: `ours_en` and `ours_ko` (the paper's
ablation/control rows live in the research repo; this release keeps the
baseline only).

## Layout

```
train_s1.py  train_s2.py  eval.py  eval_s1.py  test_s1.py    <- Hydra entry points (main() only)
models/stage1/            SpiralNet++ encoder (official/ vendored)
models/stage2/            frozen mesh encoder, bridge, Qwen3 LoRA wrapper
datasets/                 mesh_dataset (shards) + qa_dataset (QA jsonl), splits
losses/  metrics/  trainers/  evaluators/                    fact-weighted LM loss, Mask-F1,
                                                             direction/abstention metrics, DDP trainer,
                                                             unseen Set1/2/3 + structured probe evaluators
configs/                  Hydra tree; configs/experiment/*.yaml = one row each
scripts/                  bash entry points: train_s1/train_s2(+_ddp,_ko), eval_unseen, eval_set1_probe_all
tools/                    python utilities (smoke_s2, build_set1_probe, sweep_stage1_threshold)
docs/                     data preparation & training guide
DATA/                     dataset root (volume/symlink — see docs/data.md)
```

Factories (`find_{model,dataset,loss,trainer,evaluator}_def`) resolve classes
from config `name`s, so new variants are added without touching entry points.

## Requirements

CUDA GPU (Stage 2 fine-tunes Qwen3-8B with LoRA; bf16, ~1 GPU with 48 GB or
2×24 GB via DDP), PyTorch ≥ 2.5 + the pip deps in `requirements.txt`.
Everything is preinstalled in the Docker image.

## Quickstart (Docker)

```bash
cp docker.env.example .env          # DATA_DIR (easiest: the pipeline's DATA_DIR),
                                    # OUTPUT_DIR (results), CACHE_DIR (HF models)
docker compose build
docker compose run --rm train --list             # experiment rows
docker compose run --rm train --smoke ours_en    # wiring check, no LLM
docker compose run --rm train_s1                 # Stage-1 pretraining
docker compose run --rm train ours_en            # Stage-2 training
docker compose run --rm eval en                  # unseen Set1/2/3 evaluation
```

Bare metal: `pip install -r requirements.txt`, put the dataset at `./DATA`
(directory or symlink), then use the same `scripts/*.sh` directly. The CLI is
Hydra — e.g. `python train_s2.py +experiment=ours_en trainers.epochs=1`.

## Data

Easiest: set `DATA_DIR` to the **3DTongueQA construction pipeline**'s own
`DATA_DIR` — the scripts detect that layout, link the `DATA/mesh/` view
(no copying) and generate the train/val/test split automatically. Stage 2
trains on the naturalized corpus (`nat_*.jsonl`) when present and otherwise
falls back to the pipeline's template QA. The exact layout and the manual
mapping are in [docs/data.md](docs/data.md).

## Provenance

This folder is the curated release of the research training repo: diagnostics,
scratch experiments and simulation stubs were removed, machine-specific
absolute paths were normalized to the `DATA/` root, and the Docker setup was
rebuilt in the same style as the dataset pipeline release. Model, loss, data
and evaluation code paths are unchanged.
