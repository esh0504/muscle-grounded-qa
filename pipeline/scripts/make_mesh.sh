#!/usr/bin/env bash
# ============================================================================
#  make_mesh.sh [START] [END] [NPROC] — 3D tongue mesh dataset (Steps 1-3)
#
#    1. validate the Step-1 spec        (modules/steps/check_design.py)
#    2. sample the activation pool      (modules/steps/sample_pool.py, skipped
#                                        if the pool already exists)
#    3. ArtiSynth headless FEM export   (modules/artisynth/, resume-safe:
#                                        finished 1000-shards are skipped)
#
#  Defaults: whole pool, NPROC workers (env NPROC, default 2). Workers split
#  the range into 1000-aligned chunks -> no shard collisions; ~6 GB heap each.
#
#  Env: ARTISYNTH_HOME (default /opt/artisynth/artisynth_core)
#       TONGUE_MODEL   (default artisynth.models.tongue3d.StableFemMuscleTongueDemo)
#       JVM_XMX        per-worker heap (default 6g)
#       PIPELINE_CONFIG  which settings/config*.json to use (paths)
#
#  Examples:
#    scripts/make_mesh.sh              # everything
#    scripts/make_mesh.sh 0 1000 1     # pilot
# ============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

AH="${ARTISYNTH_HOME:-/opt/artisynth/artisynth_core}"
MODEL="${TONGUE_MODEL:-artisynth.models.tongue3d.StableFemMuscleTongueDemo}"
XMX="${JVM_XMX:-6g}"
HEADLESS="$ROOT/modules/artisynth/export_headless.py"
IMPL="$ROOT/modules/artisynth/export_impl.py"

# ---- 1. spec check ----------------------------------------------------------
python -m modules.steps.check_design

# ---- 2. pool (skip if present) ---------------------------------------------
POOL="$(python -c "from modules.config import path; print(path('pool_txt'))")"
OUT="$(python -c "from modules.config import path; print(path('sim_out'))")"
if [ -f "$POOL" ]; then
  echo "pool exists: $POOL (delete it to resample)"
else
  python -m modules.steps.sample_pool
fi

# ---- 3. simulate ------------------------------------------------------------
if [ ! -d "$AH/classes" ]; then
  echo "ArtiSynth classes missing at $AH (set ARTISYNTH_HOME)"
  exit 1
fi
N=$(( $(grep -vc '^#' "$POOL") - 1 ))     # pool rows (minus the index header)
START="${1:-0}"
END="${2:-$N}"
NP="${3:-${NPROC:-2}}"
echo "simulate: pool rows=$N  range=[$START,$END)  workers=$NP  out=$OUT"

run_java() {  # $1=lo  $2=hi
  java -Xmx"$XMX" \
    -Dstatic.start="$1" -Dstatic.end="$2" \
    -Dstatic.pool="$POOL" -Dstatic.out="$OUT" \
    -Dstatic.impl="$IMPL" -Dstatic.model="$MODEL" \
    -cp "$AH/classes:$AH/lib/*" \
    artisynth.core.driver.Main -noGui -model "$MODEL" -script "$HEADLESS"
}

if [ "$NP" -le 1 ]; then
  run_java "$START" "$END"
else
  mkdir -p "$OUT/log"
  SPAN=$((END - START))
  CH=$(( (SPAN + NP - 1) / NP ))
  CH=$(( (CH + 999) / 1000 * 1000 ))      # 1000-aligned chunks -> no shard race
  echo "dispatching $NP workers, chunk $CH"
  pids=()
  for i in $(seq 0 $((NP - 1))); do
    LO=$((START + i * CH)); HI=$((LO + CH)); [ "$HI" -gt "$END" ] && HI=$END
    if [ "$LO" -lt "$END" ]; then
      LOG="$OUT/log/w${i}_${LO}_${HI}.log"
      echo "  worker $i : $LO..$HI   log $LOG"
      run_java "$LO" "$HI" > "$LOG" 2>&1 &
      pids+=("$!")
      sleep 3
    fi
  done
  wait "${pids[@]}"
fi
echo "mesh dataset done -> $OUT"
