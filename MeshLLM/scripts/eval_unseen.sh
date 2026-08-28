#!/usr/bin/env bash
# 미학습 테스트셋 평가 — Set1(새 메쉬) / Set2(새 질문 family) / Set3(새 질문 형식).
# 지표 정의와 해석 기준선(상한·baseline)은 DATA/unseentest/readme.md 참고.
#
# 사용법:
#   bash scripts/eval_unseen.sh                          # en (기본)
#   bash scripts/eval_unseen.sh ko
#   bash scripts/eval_unseen.sh ko run.sets=[set1]       # 일부 세트만
#   bash scripts/eval_unseen.sh en run.limit=20          # 디버그 (세트별 레코드 상한)
#   bash scripts/eval_unseen.sh ko run.score_only=true   # 생성 없이 채점만
#   bash scripts/eval_unseen.sh en checkpoint=<path>     # 평가할 가중치 지정
#   ow 평가
#   GPU=1 bash scripts/eval_unseen.sh ko
#
# 언어는 평가 그룹(configs/evaluators/unseen{,_ko}.yaml)과 실험 row 를 함께 고른다.
# 뒤 인자는 Hydra 오버라이드로 eval.py 에 그대로 넘어간다.
set -euo pipefail
umask 000   # 생성 파일 666 / 폴더 777 — 계정 간 권한 충돌 방지
cd "$(dirname "$0")/.."

# 파이프라인 레이아웃이면 DATA/mesh 링크 구성
source scripts/ensure_data.sh
ensure_mesh_layout || exit 1

LANG_ID=en
case "${1:-}" in
  en|ko) LANG_ID="$1"; shift ;;
  -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
esac

case "$LANG_ID" in
  ko) EVAL=unseen_ko; DEFAULT_EXP=ours_ko ;;
  *)  EVAL=unseen;    DEFAULT_EXP=ours_en ;;
esac
EXP="${EXP:-$DEFAULT_EXP}"      # EXP=<이름> 으로 다른 실험 row 를 평가할 수 있다
CFG="configs/evaluators/${EVAL}.yaml"
[ -f "$CFG" ] || { echo "[ERR] 평가 설정이 없습니다: $CFG" >&2; exit 1; }
EXPCFG="configs/experiment/${EXP}.yaml"
[ -f "$EXPCFG" ] || { echo "[ERR] 그런 실험이 없습니다: $EXPCFG" >&2; exit 1; }

GPU="${GPU:-0}"
export HF_HOME="${HF_HOME:-.cache/hf}"

# 체크포인트 확인 — 없으면 학습 전 가중치로 돌아 무의미한 숫자가 나온다.
OUT=$(awk -F': *' '/^output_dir:/{split($2,a,"[ \t#]"); print a[1]; exit}' "$EXPCFG")
if [ -n "$OUT" ] && [ "$OUT" != "null" ] && [ ! -e "$OUT/mm_projector.pt" ]; then
  echo "[ERR] 학습된 체크포인트가 없습니다: $OUT/mm_projector.pt" >&2
  echo "      Stage-2 를 먼저 학습하세요 (에폭이 한 번은 끝나야 저장됩니다)." >&2
  exit 1
fi

# 기본으로 checkpoint_best 를 평가한다. 옛 configs/eval/unseen_ko.yaml 에 있던 핀을
# 대신하는 것으로, 없으면 <output_dir> 의 **마지막 에폭** 가중치가 채점된다.
# 뒤에 checkpoint=... 를 주면 Hydra 규칙대로 그것이 이긴다.
PIN=()
if [ -n "$OUT" ] && [ "$OUT" != "null" ] && [ -d "$OUT/checkpoint_best" ]; then
  PIN=(checkpoint="$OUT/checkpoint_best" run.require_weights=true)
  echo "  체크포인트: $OUT/checkpoint_best (best)"
fi

echo "=== 평가: $LANG_ID (설정 $CFG, 실험 $EXP, GPU=$GPU) ==="
CUDA_VISIBLE_DEVICES="$GPU" python eval.py +experiment="$EXP" evaluators="$EVAL" \
    "${PIN[@]}" "$@"
