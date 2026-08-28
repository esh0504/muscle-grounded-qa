#!/usr/bin/env bash
# Stage-2 한국어 학습 (DDP). 설정은 configs/experiment/ours_ko.yaml 이 단일 출처다.
#
#   bash scripts/train_s2_ko.sh                       # GPU 2장 DDP
#   NPROC=1 bash scripts/train_s2_ko.sh               # 1장만
#   bash scripts/train_s2_ko.sh trainers.epochs=1     # 뒤 인자는 Hydra 오버라이드로 그대로
#
# CUDA_VISIBLE_DEVICES=0,1 만으로는 GPU 2장을 쓰지 못한다. train_s2.py 는 모델을 단일
# 디바이스에 올리므로 torchrun 으로 프로세스를 띄워야 DDP 가 켜진다.
set -euo pipefail
umask 000   # 생성 파일 666 / 폴더 777 — 계정 간 권한 충돌 방지
cd "$(dirname "$0")/.."
export HF_HOME="${HF_HOME:-.cache/hf}"

NPROC="${NPROC:-2}"
# DDP 는 유효 배치가 GPU 수만큼 커진다. config 의 grad_accum(=8, 유효 배치 16)을
# GPU 수로 나눠 단일 GPU 실행과 같은 유효 배치를 유지한다.
# grad_accum 은 configs/experiment/ours_ko.yaml 의 trainers 블록(2칸 들여쓰기)에 있고,
# 없으면 그룹 기본값 configs/trainers/trainer_s2.yaml 을 본다.
# 값만 뽑는다 (뒤에 붙은 주석 제거)
BASE_ACC=$(awk -F': *' '/^ *grad_accum:/{split($2,a,"[ \t#]"); print a[1]; exit}' \
           configs/experiment/ours_ko.yaml configs/trainers/trainer_s2.yaml)
BASE_ACC="${BASE_ACC:-8}"
ACC=$(( BASE_ACC / NPROC )); [ "$ACC" -lt 1 ] && ACC=1

echo "=== Stage-2 ko 학습: GPU ${NPROC}장, grad_accum ${BASE_ACC} → ${ACC} (유효 배치 유지) ==="
if [ "$NPROC" -le 1 ]; then
  python train_s2.py +experiment=ours_ko "$@"
else
  torchrun --nproc_per_node="$NPROC" train_s2.py \
      +experiment=ours_ko trainers.grad_accum="$ACC" "$@"
fi
