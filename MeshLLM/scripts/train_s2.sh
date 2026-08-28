#!/usr/bin/env bash
# Stage-2 학습 — 실험 하나 = configs/experiment/*.yaml 하나 (doc/experiment.md Phase 3 의 row).
#
# 사용법:
#   bash scripts/train_s2.sh --list                  # 실험 목록과 상태
#   bash scripts/train_s2.sh --smoke ours_en         # LLM 없이 데이터·마스킹·융합 경로만
#   bash scripts/train_s2.sh ours_en                 # 학습
#   GPU=1 bash scripts/train_s2.sh ours_en
#   bash scripts/train_s2.sh ours_en trainers.epochs=1 trainers.batch_size=1  # 뒤 인자는 Hydra 오버라이드
#
# 실험 이름은 configs/experiment/<이름>.yaml 의 <이름> 이다 (= `python train_s2.py +experiment=<이름>`).
# 데이터·마스킹·모델·최적화 설정은 전부 그 yaml 이 정하고, 공통값은 그룹 기본값
# (configs/trainers/trainer_s2.yaml · configs/models/stage2_model.yaml · configs/datasets/qa_dataset.yaml) 에 있다.
# 명령줄 인자(Hydra 오버라이드)를 주면 그것이 최종 우선한다.
#
set -euo pipefail
umask 000   # 생성 파일 666 / 폴더 777 — 계정 간 권한 충돌 방지
cd "$(dirname "$0")/.."

EXPDIR=configs/experiment

list_exps() {
  printf "%-18s %-4s %-6s %s\n" 실험 phase 우선도 설명
  for f in "$EXPDIR"/*.yaml; do
    n=$(basename "$f" .yaml)
    pr=$(awk -F': *' '/^priority:/{split($2,a,"[ \t#]"); print a[1]; exit}' "$f")
    ph=$(awk -F': *' '/^phase:/{split($2,a,"[ \t#]"); print a[1]; exit}' "$f")
    # 첫 줄은 `# @package _global_` 지시자이므로 건너뛰고 그 다음 주석 줄을 설명으로 쓴다.
    desc=$(awk 'NR<=4 && /^# / && !/@package/{sub(/^# /,""); print; exit}' "$f")
    printf "%-18s %-4s %-6s %s\n" "$n" "${ph:-3}" "${pr:-?}" "$desc"
  done
}

SMOKE=0
[ "${1:-}" = "--smoke" ] && { SMOKE=1; shift; }
case "${1:-}" in
  --list|"") list_exps; exit 0 ;;
  -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
esac
NAME="$1"; shift
EXP="$EXPDIR/${NAME}.yaml"
[ -f "$EXP" ] || { echo "[ERR] 그런 실험이 없습니다: $EXP" >&2; echo; list_exps >&2; exit 1; }

GPU="${GPU:-0}"
export HF_HOME="${HF_HOME:-.cache/hf}"
OUT=$(awk -F': *' '/^output_dir:/{split($2,a,"[ \t#]"); print a[1]; exit}' "$EXP")
OUT="${OUT:-outputs/stage2/$NAME}"

# ---------- 스모크 ----------
if [ "$SMOKE" = "1" ]; then
  echo "=== Stage-2 스모크: $NAME (LLM 로드 없음) ==="
  CUDA_VISIBLE_DEVICES="$GPU" python tools/smoke_s2.py +experiment="$NAME" \
      datasets.max_records="${MAX_RECORDS:-200}" "$@"
  exit 0
fi

# ---------- 사전 점검 ----------
exec 9>/tmp/train_s2.lock
flock -n 9 || { echo "[ERR] 이미 실행 중입니다 (/tmp/train_s2.lock)." >&2; exit 1; }

# 데이터 준비: 파이프라인 레이아웃이면 DATA/mesh 링크 구성 + split 자동 생성
source scripts/ensure_data.sh
ensure_mesh_layout || exit 1
ensure_split

CKPT=$(awk -F': *' '/^ *stage1_ckpt:/{split($2,a,"[ \t#]"); print a[1]; exit}' "$EXP" \
       || true)
[ -n "$CKPT" ] || CKPT=$(awk -F': *' '/^ *stage1_ckpt:/{split($2,a,"[ \t#]"); print a[1]; exit}' configs/models/stage2_model.yaml)
INIT=$(awk -F': *' '/^ *encoder_init:/{split($2,a,"[ \t#]"); print a[1]; exit}' "$EXP")
[ -n "$INIT" ] || INIT=$(awk -F': *' '/^ *encoder_init:/{split($2,a,"[ \t#]"); print a[1]; exit}' configs/models/stage2_model.yaml)
if [ "$INIT" = "pretrained" ] && [ ! -f "$CKPT" ]; then
  echo "[ERR] Stage-1 체크포인트가 없습니다: $CKPT" >&2
  echo "      먼저: bash scripts/train_s1.sh" >&2
  exit 1
fi

BUSY=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | awk -F', ' '$2+0>1000' | wc -l)
[ "$BUSY" -eq 0 ] || {
  echo "[경고] GPU 사용 중. Qwen3-8B(bf16)는 여유 메모리가 필요합니다:" >&2
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >&2
  echo "       계속하려면 5초 안에 Ctrl-C 하지 마세요." >&2; sleep 5
}

if [ -e "$OUT/mm_projector.pt" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "[ERR] $OUT 에 이전 학습 결과가 있습니다. 덮어쓰려면 FORCE=1." >&2; exit 1
fi

mkdir -p "$OUT"
LOG="$OUT/train.log"
echo "=== Stage-2 학습: $NAME (GPU=$GPU) ==="
echo "  실험 설정: $EXP"
echo "  출력      : $OUT  (로그 $LOG)"
CUDA_VISIBLE_DEVICES="$GPU" python train_s2.py +experiment="$NAME" "$@" 2>&1 \
  | tee "$LOG" | grep -E "^\[stage2\]|^  epoch|Traceback|Error|error"
echo "=== 끝 → $OUT ==="
