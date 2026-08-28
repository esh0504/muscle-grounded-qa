# Training & evaluation guide

The CLI is **Hydra** everywhere: config groups live under `configs/`, an
experiment row is one overlay in `configs/experiment/<name>.yaml`, and any
trailing `key=value` args are Hydra overrides with final priority.
`python train_s2.py +experiment=ours_en --cfg job` prints the merged config
without touching a GPU.

## Stage 1 — muscle-regression pretraining

```bash
bash scripts/train_s1.sh --smoke      # 50-step wiring check
bash scripts/train_s1.sh             # the mesh encoder (frozen and reused by Stage 2)
EPOCHS=300 GPU=1 bash scripts/train_s1.sh
```

Hyperparameters come from `configs/trainers/trainer_s1.yaml` (single source);
`checkpoint_best.pt` is refreshed on val, so Ctrl-C keeps the best model.
Evaluate with `python eval_s1.py` / threshold sweep via
`tools/sweep_stage1_threshold.py`.

## Stage 2 — mesh-conditioned LLM

```bash
bash scripts/train_s2.sh --smoke ours_en           # data/masking/fusion wiring, no LLM
bash scripts/train_s2.sh ours_en                   # single GPU
bash scripts/train_s2_ddp.sh ours_en trainers.epochs=1    # 2-GPU DDP
bash scripts/train_s2_ko.sh                        # Korean instantiation
```

Key mechanisms (see the config comments in `configs/datasets/qa_dataset.yaml`):
fact-span loss weighting (`span_select_ratio` / `target_fact_share`),
context-span masking against copy shortcuts (`context_mask_prob`), turn-wise
samples to avoid future-question leakage, and `enable_thinking: false` to
match the Qwen3 generation prompt at eval time.

## Evaluation

```bash
bash scripts/eval_unseen.sh en                     # unseen Set1/2/3 (metrics + report)
bash scripts/eval_unseen.sh ko run.sets=[set1]
bash scripts/eval_unseen.sh en run.limit=20        # debug
bash scripts/eval_set1_probe_all.sh                # structured Set-1 probe across rows
```

Metrics live in `metrics/` (Mask-F1, direction/sign accuracy, abstention F1,
monotonicity, muscle regression, numeric ΔF) and are aggregated by
`evaluators/unseen.py` into per-set reports under the experiment's
`output_dir` (default `outputs/stage2/<row>/`).

## Outputs & caches

Checkpoints, eval reports and Hydra run logs: `outputs/` — in Docker this is
the `${OUTPUT_DIR}` volume, so point OUTPUT_DIR in `.env` at a disk with room
(override per run with `output_dir=`). Hugging Face model cache: `.cache/hf`
(`HF_HOME`; the Docker volume `${CACHE_DIR}` keeps it across containers —
Qwen3-8B is ~16 GB on first download).

## Tests

`pytest tests/` — import/shape/config-parity checks; no GPU, no LLM download.
