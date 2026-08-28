# settings/ — everything the user configures

Every knob a user touches lives in THIS folder; the code (`scripts/`,
`modules/`) only reads these values.

| File | Contents | When to edit |
|---|---|---|
| `docker.env.example` → copy to root `.env` | Docker: `DATA_DIR` (required), image/container names, NPROC, JVM_XMX | when using Docker |
| `config.json` | bare-metal paths: `sim_out` (simulation disk), pool/index/qa paths (`rest_index` defaults to `auto`) | when running directly on Windows |
| `config.docker.json` | container paths (all under `/data`) | usually never |
| `anchors.yaml` | **12 vowel centers + 8 consonant rules** + AMAX/budgets (Step-1 anchor spec) | when changing the experimental design |
| `sampling.yaml` | **pool total/seed + section ratios** (anchor/neighbor/effort/spacefill), levels/grids | when changing the experimental design |
| `centers.csv` | QA anchor-target table (Step-5 prescriptive / target-directed questions) | when changing the experimental design |
| `templates_ko.yaml` / `templates_en.yaml` | **full QA templates**: question/answer wording, muscle display names, direction/region words, mask_spans keywords, output filenames | when rewording QA or adding a language |

## Minimal setup

**Docker** — one value: `DATA_DIR` in `.env` (~1 TB disk for the full 295k run):

```bash
cp settings/docker.env.example .env             # at the pipeline root
docker compose build && docker compose run --rm mesh
```

**Windows bare metal** — two values: `sim_out` in `config.json` + the
`ARTISYNTH_HOME` environment variable (used by `modules\artisynth\run_headless.bat`;
the shipped defaults match the original machine).

## Reproducibility notes

Changing `anchors.yaml` / `sampling.yaml` / `centers.csv` /
`templates_*.yaml` produces a DIFFERENT pool/QA than the released dataset —
leave them untouched for exact reproduction. The rest posture is pinned at pool
index 0 by default (`rest_first: true` in `sampling.yaml`) and
`rest_index: auto` finds it from pool_meta automatically, so nothing needs
updating by hand. To reproduce the released pool byte-for-byte (295,157 rows,
rest at index 290609), set `rest_first: false`.

After editing the design files, run `python -m modules.steps.check_design`
once (budgets, bands, ratios, centers consistency). With Docker, code and
settings are baked into the image — rebuild with `docker compose build` after
edits (`.env` is the exception: recreate the container, no rebuild needed).
QA generation is resume-based, so delete existing QA shard files before
regenerating with changed templates.

## Naming customization (optional, .env)

| Variable | Default | What |
|---|---|---|
| `COMPOSE_PROJECT_NAME` | `tongueqa` | compose project (Docker Desktop group) |
| `IMAGE_NAME` / `IMAGE_TAG` | `tongueqa/pipeline` / `latest` | image name/tag |
| `CONTAINER_NAME` | `tongueqa_pipeline` | the persistent `workspace` container |
