# Docker environment

Self-contained image for the whole construction pipeline: ArtiSynth (core +
models, compiled at pinnable refs) + Python (numpy/pandas) + this repository
baked at `/workspace`. The only external thing is the `/data` volume:

```
/data/mesh/pool.txt, pool_meta.csv        Step 2
/data/mesh/{topology.obj,verts,nodes,meta} Step 3   (~1 TB at full 295k scale)
/data/index/                              Step 5a-c
/data/qa/{physics,features}_{ko,en}/      Step 5d-e
```

## Quickstart

```bash
cd pipeline                                   # repo root
cp settings/docker.env.example .env            # edit DATA_DIR (big disk!)
echo "DOCKER_UID=$(id -u)" >> .env              # shared server: write files as YOUR
echo "DOCKER_GID=$(id -g)" >> .env              # account, not root
mkdir -p <DATA_DIR>                             # create it yourself (root-owned otherwise)
docker compose build
docker compose run --rm mesh                   # steps 1-3 (spec check + pool + FEM meshes)
docker compose run --rm qa                     # steps 4-5 (indexes + KO/EN QA)
```

`mesh` accepts an explicit range/worker override
(`run --rm mesh 0 295157 4`) and is resume-safe — re-run the same command
after a crash and finished shards are skipped. `qa` is likewise shard-resumable.
For an interactive shell: `up -d workspace` + `exec workspace bash`.

Container config is `settings/config.docker.json` (via `PIPELINE_CONFIG`);
`settings/config.json` remains the bare-metal default. Run docker compose from
the pipeline root — it picks up `.env` and `docker-compose.yml` there.

## Release checklist

- Build with pinned refs and record them alongside the dataset:
  `build --build-arg ARTISYNTH_REF=<commit> --build-arg MODELS_REF=<commit>`
  (the image also writes the resolved commits to `/opt/artisynth/REFS`,
  printed at container start).
- The image bakes the code at build time — rebuild after editing pipeline code
  or settings. For iteration, use DEV mode instead:
  `docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d workspace`
  (bind-mounts the repo over `/workspace`, so edits apply immediately without
  rebuilding). Ship/release with the baked image, not dev mode.
- Native solver libraries are x86_64; on Apple Silicon enable the
  `platform: linux/amd64` line in the compose file.
- `NPROC` workers need ~6 GB heap each (`JVM_XMX`) plus a core.

## Note on provenance

The released 295,157-mesh dataset was produced on Windows with a local
ArtiSynth checkout. This image reproduces the pipeline end-to-end, but an
ArtiSynth revision different from that checkout can shift solver results
slightly — pin the matching refs when exact reproduction matters.
