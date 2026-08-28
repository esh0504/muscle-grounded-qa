#!/usr/bin/env bash
# Stage-1 학습 — SpiralNet++ 인코더로 mesh 변위 → 근육 활성 11개 (DATA/mesh).
#
# 사용법:
#   bash scripts/train_s1.sh --smoke            # 50 step 만 돌려 배선 확인 (wandb 끔, 결과 버림)
#   bash scripts/train_s1.sh                    # 인코더 학습 (Stage-2 가 동결 재사용)
#   EPOCHS=300 bash scripts/train_s1.sh         # 에폭 수 임시 변경 (안 주면 trainer_s1.yaml 값)
#   GPU=1 bash scripts/train_s1.sh              # 다른 GPU 로
#   FORCE=1 bash scripts/train_s1.sh            # 기존 출력 폴더 덮어쓰기 허용
#   bash scripts/train_s1.sh trainers.batch_size=128 trainers.lr=5e-4        # 뒤에 붙이면 hydra 오버라이드
#
# 알아둘 것:
#   - 에폭·배치·lr 등 학습 설정은 configs/trainers/trainer_s1.yaml 이 단일 출처다.
#     이 스크립트는 EPOCHS 를 명시적으로 준 경우에만 그 값을 덮어쓴다.
#   - lr 이 0.99^epoch 라 500 에폭이면 1e-5 밑이다. 그 뒤로는 사실상 학습이 진행되지 않는다.
#   - val 기준으로 checkpoint_best.pt 를 계속 갱신하므로 Ctrl-C 로 끊어도 최고 성능은 남는다.
set -euo pipefail
umask 000   # 생성 파일 666 / 폴더 777 — 계정 간 권한 충돌 방지
cd "$(dirname "$0")/.."

SMOKE=0
case "${1:-}" in
  --smoke) SMOKE=1; shift ;;
  "" ) ;;
  -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
esac

# 에폭은 configs/trainers/trainer_s1.yaml 이 정한다. EPOCHS 를 준 경우에만 그것을 덮어쓴다.
# (예전엔 여기서 항상 오버라이드를 넘겨서 config 를 고쳐도 반영이 안 됐다.)
EPOCHS="${EPOCHS:-}"
CFG_EPOCHS=$(awk '/^epochs:/{print $2; exit}' configs/trainers/trainer_s1.yaml)
EFF_EPOCHS="${EPOCHS:-$CFG_EPOCHS}"
GPU="${GPU:-0}"
OUT_ROOT="${OUT_ROOT:-outputs}"
EXTRA=("$@")   # 나머지 인자는 hydra 로 그대로 넘긴다

# ---------- 사전 점검 ----------
exec 9>/tmp/train_s1.lock
flock -n 9 || { echo "[ERR] 이미 실행 중입니다 (/tmp/train_s1.lock). 동시 실행은 서로 느리게 만든다." >&2; exit 1; }

# 데이터 준비: 파이프라인 레이아웃이면 DATA/mesh 링크 구성 + split 자동 생성
source scripts/ensure_data.sh
ensure_mesh_layout || exit 1
ensure_split

BUSY=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | awk -F', ' '$2+0>1000' | wc -l)
[ "$BUSY" -eq 0 ] || {
  echo "[경고] GPU 를 이미 쓰는 프로세스가 있습니다:" >&2
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv >&2
  echo "       그래도 계속하려면 5초 안에 Ctrl-C 하지 마세요." >&2
  sleep 5
}

# ---------- 한 판 돌리기 ----------
run_one() {
  local tag="$1" model="$2"
  shift 2
  local out="$OUT_ROOT/${tag}"
  local -a hy=(models="$model" "$@")   # 나머지 = 이 row 전용 hydra 오버라이드
  [ -n "$EPOCHS" ] && hy+=(trainers.epochs="$EPOCHS")

  if [ "$SMOKE" = "1" ]; then
    out="$OUT_ROOT/${tag}_smoke"
    hy=(models="$model" trainers.epochs=1 trainers.max_steps=50 trainers.wandb.enabled=false)
    rm -rf "$out"
  elif compgen -G "$out/checkpoint_*.pt" >/dev/null && [ "${FORCE:-0}" != "1" ]; then
    # 트레이너는 이어하기를 지원하지 않는다. 그냥 두면 이전 결과를 덮어쓴다.
    echo "[ERR] $out 에 이전 학습 결과가 있습니다. 덮어쓰려면 FORCE=1, 남기려면 OUT_ROOT 를 바꾸세요." >&2
    return 1
  fi

  mkdir -p "$out"
  local log="$out/train.log"
  local ep="$EFF_EPOCHS"; [ "$SMOKE" = "1" ] && ep=1
  echo "[$(date +%H:%M:%S)] $tag 시작 (GPU=$GPU, epochs=$ep) → $log"
  CUDA_VISIBLE_DEVICES="$GPU" python train_s1.py \
      "${hy[@]}" \
      trainers.output_dir="$out" \
      trainers.wandb.name="${tag}" \
      "${EXTRA[@]}" 2>&1 | tee "$log" | grep -E "^\[Stage-1\]|^\[TrainerS1\]|^epoch|View run|Traceback|Error"
  echo "[$(date +%H:%M:%S)] $tag 완료 → $out/checkpoint_best.pt"
  grep -E "^epoch" "$log" | awk '
    {split($2,e,"/"); for(i=1;i<=NF;i++) if($i ~ /^val_loss=/){split($i,a,"=");
      if(n==0 || a[2]+0<best){best=a[2]+0; be=e[1]} n++}}
    END{if(n) printf("         최고 val_loss=%.6f (epoch %s), 총 %d epoch\n", best, be, n)}'
}

if [ "$SMOKE" = "1" ]; then
  echo "=== Stage-1 스모크 (50 step, wandb 끔, 결과는 outputs/stage1_smoke 로 버림) ==="
else
  if [ -n "$EPOCHS" ]; then
    echo "=== Stage-1 학습 (epochs=$EFF_EPOCHS [EPOCHS 로 지정], GPU=$GPU) ==="
  else
    echo "=== Stage-1 학습 (epochs=$EFF_EPOCHS [trainer_s1.yaml], GPU=$GPU) ==="
  fi
fi
run_one stage1 stage1_model
