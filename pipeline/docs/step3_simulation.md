# Step 3 — FEM Simulation & Validity Screening (ArtiSynth, headless)

Maps every pool row to a settled static tongue mesh with the ArtiSynth Badin FEM
model (`StableFemMuscleTongueDemo`): minimum-jerk activation ramp (1.0 s) →
adaptive settle (0.2–1.5 s, early exit once max node speed < 1e-3 m/s) →
validity watchdog (NaN, inverted elements, per-element & global volume ratio,
settle/ramp velocity) → labels `VALID / MARGINAL / INVALID_PHYSICAL /
FAILED_NUMERICAL`. Failures are kept (FailureBank), only `VALID` is used
downstream.

Files:

- `step3_export_impl.py` — the exporter (ArtiSynth **Jython, Python-2 syntax
  on purpose**; do not port to Python 3). All simulation constants and their
  rationale are documented inline (ramp-vs-settle, ramp_sweep evidence).
- `step3_export_headless.py` — `-noGui` entry point; reads everything from
  `-Dstatic.*` system properties.
- `step3_run_headless.bat` — worker launcher / dispatcher (Windows).
- `step3_run_headless.sh` — the same launcher for Linux / the project's Docker image.

## Run (Windows machine with compiled ArtiSynth)

```bat
set ARTISYNTH_HOME=C:\Users\d11\artisynth\artisynth_core   (default; env var)
step3_run_headless.bat 0 295157 4
step3_run_headless.bat 0 295157 4 D:\static_300k ..\outputs\pool.txt
```

- NPROC workers over 1000-aligned disjoint ranges → no shard collisions;
  ~6 GB RAM + 1 core per worker.
- **Resume**: re-run the same command; a shard counts as done only when its
  `meta/shard_*.csv` exists, so a killed run redoes at most one shard per worker.
- Never drive the full batch from the ArtiSynth GUI console — the timeline
  redraw races with probe add/remove (`ConcurrentModificationException`).
- Useful GUI-console helpers (load the model, then run the script once):
  `probe_run(20)` for throughput + watchdog sanity, `ramp_sweep()` before ever
  touching `RAMP_T`.

## Run (Docker image from the research repo)

The existing `xai/mri_transfer` image already contains everything Step 3 needs
(ArtiSynth core+models built at `/opt/artisynth/artisynth_core`, Java, Python).
Inside the container:

```bash
scripts/run_headless.sh 0 295157 4 /data/mesh
```

Notes when reusing that image:

- Make sure this `pipeline/` folder is inside a mounted volume (the compose file
  mounts the repo root at `/workspace`; add a volume line or place `pipeline/`
  under the mounted directory).
- The image's env default `TONGUE_MODEL` is `HexTongueDemo`; `step3_run_headless.sh`
  overrides it to `StableFemMuscleTongueDemo` — keep that override.
- `pip install pandas` once (the image's requirements.txt has numpy but not
  pandas, which Step 5 needs).
- The image clones ArtiSynth `master` at build time; for strict bit-level
  reproduction pin the same artisynth_core/artisynth_models commits that were
  used for the released dataset.

Output layout is documented in the impl header and read by `modules/mesh_io.py`.
