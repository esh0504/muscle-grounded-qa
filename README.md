# Muscle-Grounded QA

Simulator-grounded construction of **verifiable muscle-grounded QA** from 3D
tongue meshes — the framework, its **3DTongueQA** instantiation, and the
**MeshLLM** validation baselines, from

> *A Simulator-Grounded Framework for Constructing Verifiable Muscle-Grounded
> QA from 3D Tongue Meshes* (ICASSP submission).

Controlled 11-D muscle activations are mapped through the ArtiSynth Badin FEM
tongue model to validity-screened 3D meshes, converted into structured
biomechanical records, and rendered as deterministic, record-verifiable QA
(Korean/English). A two-stage baseline then tests that the supervision is
learnable and geometry-grounded.

```
pipeline/   dataset construction (Steps 1-5)
            sampling & anchor design -> muscle pool -> FEM simulation +
            screening -> QA templates -> QA generation
            CPU · Java/ArtiSynth Docker image · settings/-driven

MeshLLM/    validation baselines
            Stage 1: SpiralNet++ mesh->muscle regression (muscle-aware pretraining)
            Stage 2: frozen mesh encoder -> bridge -> Qwen3-8B (LoRA) unified QA
            + unseen-set evaluators & span-level metrics
            GPU · PyTorch/CUDA Docker image · Hydra configs
```

## Quickstart

Each component is self-contained with its own Docker image; they connect
through a shared data volume.

```bash
# 1) build the dataset (pipeline/README.md)
cd pipeline
cp settings/docker.env.example .env        # set DATA_DIR (~1 TB for the full run)
docker compose build
docker compose run --rm mesh               # steps 1-3: spec -> pool -> FEM meshes
docker compose run --rm qa                 # steps 4-5: indexes -> KO/EN QA

# 2) train & evaluate the baselines (MeshLLM/README.md)
cd ../MeshLLM
cp docker.env.example .env                 # DATA_DIR built per MeshLLM/docs/data.md
docker compose build
docker compose run --rm train_s1      # Stage-1 pretraining
docker compose run --rm train ours_en      # Stage-2 training
docker compose run --rm eval en            # unseen Set1/2/3 evaluation
```

The mapping from pipeline outputs to the training data layout (plus the
train/val/test split) is documented in `MeshLLM/docs/data.md`.

## What "verifiable" means here

Every gold answer is rendered deterministically from structured simulator
records, and every factual token (muscle names, numbers, directions, regions)
carries a character-level `mask_spans` annotation. Language naturalization
changes surface form only and is re-verified against the records — so any QA
pair can be audited back to the controlled physical input that produced it.

## Reproducibility

- The pool sampler and QA generators are deterministic given the settings in
  `pipeline/settings/` (released pool: total=300000, seed=0 -> 295,157 rows;
  see `pipeline/settings/README.md` for the `rest_first` note).
- Pin the ArtiSynth refs when building the pipeline image
  (`--build-arg ARTISYNTH_REF=... MODELS_REF=...`); the resolved commits are
  recorded in the image.
- Baseline experiment rows are one Hydra overlay each
  (`MeshLLM/configs/experiment/*.yaml`).

## Citation

```bibtex
@inproceedings{eum2027muscle,
  title     = {A Simulator-Grounded Framework for Constructing Verifiable
               Muscle-Grounded QA from 3D Tongue Meshes},
  author    = {Eum, Seungho and Park, Unsang},
  booktitle = {ICASSP},
  year      = {2027}
}
```
