#!/usr/bin/env bash
# Stage-2 실험 row 를 DDP 로 돌린다.
#
#   bash scripts/train_s2_ddp.sh ours_en                       # GPU 2장
#   NPROC=1 bash scripts/train_s2_ddp.sh ours_en               # 1장
#   bash scripts/train_s2_ddp.sh ours_en trainers.epochs=2     # 뒤 인자는 Hydra 오버라이드로 그대로
#
# scripts/train_s2.sh 를 쓰지 않는 이유:
#   ① flock 을 잡아 두 번째 동시 실행이 죽는다
#   ② torchrun 을 쓰지 않아 단일 GPU 전용이다
#   ③ DDP 는 유효 배치가 GPU 수만큼 커진다. grad_accum 을 GPU 수로 나누지 않으면
#      유효 배치가 16 이 아니라 32 가 되어 **다른 실험**이 된다. 이 스크립트가 자동으로 나눈다.
set -euo pipefail
umask 000   # 생성 파일 666 / 폴더 777 — 계정 간 권한 충돌 방지
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-.cache/hf}"

EXP="${1:?사용법: bash scripts/train_s2_ddp.sh <실험이름> [Hydra 오버라이드...]}"
shift || true
CFG="configs/experiment/${EXP}.yaml"
[ -f "$CFG" ] || { echo "설정이 없다: $CFG"; ls configs/experiment/*.yaml; exit 1; }

# 코드가 아직 없는 row(zeroshot_vlm)는 configs/experiment/ 로 이식하지 않았다.
# 그래서 위의 파일 존재 확인이 예전 `implemented: false` 검사를 대신한다.

NPROC="${NPROC:-2}"
# torchrun 의 rendezvous 포트. **이미 다른 DDP run 이 돌고 있으면 반드시 바꿔야 한다** —
# 기본 29500 을 그대로 쓰면 두 번째 run 이 EADDRINUSE 로 즉사한다.
#   MASTER_PORT=29520 bash scripts/train_s2_ddp.sh ours_en
MASTER_PORT="${MASTER_PORT:-29500}"
# grad_accum 은 configs/experiment/<exp>.yaml 의 trainers 블록(2칸 들여쓰기)에 있고, 없으면
# 그룹 기본값 configs/trainers/trainer_s2.yaml 을 본다. 뒤에 붙은 주석은 떼어낸다.
BASE_ACC=$(awk -F': *' '/^ *grad_accum:/{split($2,a,"[ \t#]"); print a[1]; exit}' \
           "$CFG" configs/trainers/trainer_s2.yaml)
BASE_ACC="${BASE_ACC:-8}"
ACC=$(( BASE_ACC / NPROC )); [ "$ACC" -lt 1 ] && ACC=1

OUT=$(awk -F': *' '/^output_dir:/{split($2,a,"[ \t#]"); print a[1]; exit}' "$CFG")
echo "=== ${EXP}: GPU ${NPROC}장, grad_accum ${BASE_ACC} → ${ACC} (유효 배치 유지) ==="
echo "=== out: ${OUT} ==="
if [ -f "${OUT}/mm_projector.pt" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "[중단] 이미 학습된 결과가 있다: ${OUT}/mm_projector.pt  (FORCE=1 로 덮어쓰기)"; exit 1
fi

if [ "$NPROC" -le 1 ]; then
  exec python train_s2.py +experiment="$EXP" "$@"
else
  exec torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" train_s2.py \
       +experiment="$EXP" trainers.grad_accum="$ACC" "$@"
fi
