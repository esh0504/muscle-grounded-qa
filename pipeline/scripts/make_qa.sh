#!/usr/bin/env bash
# ============================================================================
#  make_qa.sh — QA dataset from the simulated meshes (Steps 4-5)
#
#    indexes : build_index -> region_disp -> extract_features
#    QA      : physics chain + feature A1/B3, per language (Step-4 templates)
#
#  Resume-safe: finished QA shard files are skipped; re-run after a crash.
#
#  Env: LANGS         languages to generate (default "ko en")
#       FEAT_WORKERS  feature-extraction processes (default 2)
#       PIPELINE_CONFIG  which settings/config*.json to use (paths)
#
#  Example: scripts/make_qa.sh
# ============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

POOL="$(python -c "from modules.config import path; print(path('pool_txt'))")"
SIM="$(python -c "from modules.config import path; print(path('sim_out'))")"
QA="$(python -c "from modules.config import path; print(path('qa_out'))")"
if [ ! -f "$POOL" ]; then
  echo "pool not found: $POOL — run scripts/make_mesh.sh first"
  exit 1
fi
if ! ls "$SIM"/meta/shard_*.csv >/dev/null 2>&1; then
  echo "no simulation meta under $SIM — run scripts/make_mesh.sh first"
  exit 1
fi

N=$(( $(grep -vc '^#' "$POOL") - 1 ))
LANGS="${LANGS:-ko en}"
echo "qa: pool rows=$N  langs=$LANGS"

python -m modules.steps.build_index
python -m modules.steps.region_disp
python -m modules.steps.extract_features --workers "${FEAT_WORKERS:-2}"

for L in $LANGS; do
  python -m modules.steps.gen_physics_qa --lang "$L" 0 "$N"
  python -m modules.steps.gen_feature_qa --lang "$L" A1 0 "$N"
  python -m modules.steps.gen_feature_qa --lang "$L" B3
done
echo "QA done -> $QA"
