#!/usr/bin/env bash
# Set-1 probe 일괄 평가 — baseline rows (ours_en / ours_ko)
#
# 사용:
#   bash scripts/eval_set1_probe_all.sh              # 전부 재예측 (GPU 0)
#   GPUS=0,1 bash scripts/eval_set1_probe_all.sh     # 2장에 나눠 병렬 재예측
#   bash scripts/eval_set1_probe_all.sh --score-only # 재예측 없이 preds 재채점·pred/ 갱신
#   bash scripts/eval_set1_probe_all.sh --skip-done  # pred/ 완료된 row 만 건너뜀
#   bash scripts/eval_set1_probe_all.sh ours_ko      # 일부만
#   bash scripts/eval_set1_probe_all.sh --dry-run
#
set -euo pipefail
umask 000   # 생성 파일 666 / 폴더 777 — 계정 간 권한 충돌 방지
cd "$(dirname "$0")/.."

# 파이프라인 레이아웃이면 DATA/mesh 링크 구성
source scripts/ensure_data.sh
ensure_mesh_layout || exit 1

export HF_HOME="${HF_HOME:-.cache/hf}"

SCORE_ONLY=false
SKIP_DONE=false
DRY_RUN=false
EXTRA=()
SELECT=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --score-only) SCORE_ONLY=true; shift ;;
    --force)      shift ;;   # 하위 호환: 기본이 이미 재예측이라 no-op
    --skip-done)  SKIP_DONE=true; shift ;;
    --dry-run)    DRY_RUN=true; shift ;;
    -h|--help)
      sed -n '2,24p' "$0"; exit 0 ;;
    --) shift; EXTRA+=("$@"); break ;;
    -*)
      echo "[ERR] unknown flag: $1" >&2; exit 1 ;;
    *)
      SELECT+=("$1"); shift ;;
  esac
done

# name | experiment | evaluators | extra hydra overrides (comma-separated, optional)
JOBS=(
  "ours_en|ours_en|set1_probe|"
  "ours_ko|ours_ko|set1_probe_ko|"
)

GPUS_CSV="${GPUS:-${GPU:-0}}"
IFS=',' read -r -a GPU_ARR <<< "$GPUS_CSV"
N_GPU=${#GPU_ARR[@]}

ROOT_DATA="${ROOT_DATA:-outputs/stage2}"
LOG_DIR="${LOG_DIR:-$ROOT_DATA/_eval_logs}"
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_TSV="$LOG_DIR/set1_summary_${STAMP}.tsv"

_want() {
  local name="$1"
  if [[ ${#SELECT[@]} -eq 0 ]]; then
    return 0
  fi
  local s
  for s in "${SELECT[@]}"; do
    [[ "$s" == "$name" ]] && return 0
  done
  return 1
}

_out_dir_of() {
  # 실험 yaml 의 output_dir
  local exp="$1"
  awk -F': *' '/^output_dir:/{split($2,a,"[ \t#]"); print a[1]; exit}' \
    "configs/experiment/${exp}.yaml"
}

_eval_dir_of() {
  local name="$1" exp="$2"
  echo "$(_out_dir_of "$exp")/eval"
}

_done() {
  # pred 레이아웃 + summary 있으면 완료로 본다
  local edir="$1"
  [[ -f "$edir/pred/summary.json" \
     && -f "$edir/pred/auto_metric_json/muscle_set.json" \
     && -f "$edir/pred/judge_metric_json/gpt/mesh_preds.json" \
     && -f "$edir/pred/judge_metric_json/gpt/reference_answer.json" ]]
}

_build_cmd() {
  # stdout: 한 줄 hydra 명령 (python eval.py ...)
  local name="$1" exp="$2" ev="$3" kind="$4"
  local edir; edir="$(_eval_dir_of "$name" "$exp")"
  local args=(python eval.py "+experiment=${exp}" "evaluators=${ev}"
              "evaluators.output_dir=${edir}")

  if [[ "$SCORE_ONLY" == true ]]; then
    args+=(run.score_only=true)
  fi

  if [[ ${#EXTRA[@]} -gt 0 ]]; then
    args+=("${EXTRA[@]}")
  fi
  printf '%q ' "${args[@]}"
  printf '\n'
}

echo "=== Set-1 probe batch ==="
echo "  GPUS=$GPUS_CSV  score_only=$SCORE_ONLY  skip_done=$SKIP_DONE  dry_run=$DRY_RUN"
echo "  mode=$([[ "$SCORE_ONLY" == true ]] && echo '재채점만' || echo '재예측+채점+렌더')"
echo "  log_dir=$LOG_DIR"
echo

declare -a RUN_NAMES=()
declare -a RUN_CMDS=()
declare -a RUN_LOGS=()
declare -a RUN_EDIRS=()

for spec in "${JOBS[@]}"; do
  IFS='|' read -r name exp ev kind <<< "$spec"
  _want "$name" || continue
  edir="$(_eval_dir_of "$name" "$exp")"

  # 기본은 5개 전부 재예측. --skip-done 일 때만 완료 row 건너뜀.
  if [[ "$SKIP_DONE" == true ]] && _done "$edir"; then
    echo "[skip] $name  already done → $edir/pred/"
    continue
  fi

  # --score-only 인데 preds 없으면 풀 런으로 승격. 기본은 항상 재예측.
  local_score="$SCORE_ONLY"
  if [[ "$SCORE_ONLY" == true && ! -f "$edir/preds.jsonl" ]]; then
    echo "[info] $name  preds.jsonl 없음 → 전체 추론으로 진행"
    local_score=false
  fi

  # _build_cmd 는 전역 SCORE_ONLY 를 보므로 잠깐 덮어쓴다
  save_so=$SCORE_ONLY
  SCORE_ONLY=$local_score
  cmd="$(_build_cmd "$name" "$exp" "$ev" "$kind")"
  SCORE_ONLY=$save_so

  log="$LOG_DIR/${name}_${STAMP}.log"
  RUN_NAMES+=("$name")
  RUN_CMDS+=("$cmd")
  RUN_LOGS+=("$log")
  RUN_EDIRS+=("$edir")
  echo "[queue] $name"
  echo "        $cmd"
  echo "        log=$log"
done

if [[ ${#RUN_NAMES[@]} -eq 0 ]]; then
  echo "[done] 돌릴 job 없음 (전부 skip 이거나 선택 집합이 비었음)"
else
  if [[ "$DRY_RUN" == true ]]; then
    echo "[dry-run] ${#RUN_NAMES[@]} jobs queued — 실행 안 함"
  else
    # GPU 라운드로빈 병렬. N_GPU=1 이면 순차와 동일.
    # set -e 아래 ((…))·빈 배열 대입에 안 터지도록 조심해서 대기한다.
    declare -a PIDS=()
    declare -a PID_NAMES=()
    declare -a PID_LOGS=()
    fail=0

    _reap_finished() {
      local i
      local -a keep_pids=() keep_names=() keep_logs=()
      for i in "${!PIDS[@]}"; do
        if kill -0 "${PIDS[$i]}" 2>/dev/null; then
          keep_pids+=("${PIDS[$i]}")
          keep_names+=("${PID_NAMES[$i]}")
          keep_logs+=("${PID_LOGS[$i]}")
        else
          if wait "${PIDS[$i]}"; then
            echo "[ok]   ${PID_NAMES[$i]}"
          else
            echo "[FAIL] ${PID_NAMES[$i]}  (see ${PID_LOGS[$i]})" >&2
            fail=1
          fi
        fi
      done
      if ((${#keep_pids[@]})); then
        PIDS=("${keep_pids[@]}")
        PID_NAMES=("${keep_names[@]}")
        PID_LOGS=("${keep_logs[@]}")
      else
        PIDS=()
        PID_NAMES=()
        PID_LOGS=()
      fi
    }

    _wait_slot() {
      while (( ${#PIDS[@]} >= N_GPU )); do
        _reap_finished
        if (( ${#PIDS[@]} >= N_GPU )); then
          sleep 5
        fi
      done
    }

    for i in "${!RUN_NAMES[@]}"; do
      _wait_slot
      gpu="${GPU_ARR[$((i % N_GPU))]}"
      name="${RUN_NAMES[$i]}"
      cmd="${RUN_CMDS[$i]}"
      log="${RUN_LOGS[$i]}"
      echo "[run]  $name  on GPU $gpu"
      (
        export CUDA_VISIBLE_DEVICES="$gpu"
        eval "$cmd"
      ) >"$log" 2>&1 &
      PIDS+=("$!")
      PID_NAMES+=("$name")
      PID_LOGS+=("$log")
    done

    while (( ${#PIDS[@]} > 0 )); do
      _reap_finished
      if (( ${#PIDS[@]} > 0 )); then
        sleep 2
      fi
    done

    if (( fail )); then
      echo "[ERR] 일부 job 실패" >&2
      exit 1
    fi
  fi
fi

# ---- 요약 테이블 ----
echo
echo "=== headline summary ==="
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  Model "Muscle_F1" "Value_acc" "Direction_acc" "Abstention_F1" "Delta_shuf" path \
  | tee "$SUMMARY_TSV"

for spec in "${JOBS[@]}"; do
  IFS='|' read -r name exp ev kind <<< "$spec"
  _want "$name" || continue
  edir="$(_eval_dir_of "$name" "$exp")"
  sum="$edir/pred/summary.json"
  [[ -f "$sum" ]] || sum="$edir/summary.json"
  if [[ ! -f "$sum" ]]; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" - - - - - "$edir" | tee -a "$SUMMARY_TSV"
    continue
  fi
  python - "$name" "$sum" "$edir" <<'PY' | tee -a "$SUMMARY_TSV"
import json, sys
name, path, edir = sys.argv[1], sys.argv[2], sys.argv[3]
h = json.loads(open(path, encoding="utf-8").read())
def fmt(k):
    v = h.get(k)
    if v is None: return "-"
    if isinstance(v, float): return f"{v:.4f}"
    return str(v)
print("\t".join([
    name,
    fmt("Muscle F1"), fmt("Value acc"), fmt("Direction acc"),
    fmt("Abstention F1"), fmt("Δ_shuf"), edir,
]))
PY
done

echo
echo "TSV → $SUMMARY_TSV"
echo "각 row 산출: <eval>/pred/{render,auto_metric_json,judge_metric_json}"
echo "GPT 업로드: imgs_preds.json / mesh_preds.json 만 (reference_answer.json 제외)"
