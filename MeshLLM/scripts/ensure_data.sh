#!/usr/bin/env bash
# ensure_data.sh — train/eval 스크립트가 source 해서 쓰는 데이터 준비 공통부.
#
# 1) DATA 가 파이프라인 DATA_DIR 그대로면 (static_300k/ + index/ + outputs/)
#    DATA/mesh/ 뷰를 **상대 심볼릭 링크**로 자동 구성한다 — 복사 없음.
#    같은 폴더 트리 안의 상대 링크라 호스트에서도 컨테이너에서도 유효하다.
# 2) split(train/val/test.txt)이 없으면 생성한다.
#
# 이미 DATA/mesh 를 직접 구성해뒀으면 (docs/data.md 의 복사 방식) 아무것도 안 한다.

ensure_mesh_layout() {
  # 새 파이프라인 레이아웃 (mesh/ 에 pool+메쉬 일체): meta_all.csv 만 없으면 index/ 에서 링크
  if [ -e DATA/mesh/topology.obj ] && [ ! -e DATA/mesh/meta_all.csv ] && [ -e DATA/index/meta_all.csv ]; then
    ln -sfn ../index/meta_all.csv DATA/mesh/meta_all.csv
  fi

  # 옛 파이프라인 레이아웃 (static_300k/ + index/ + outputs/): mesh/ 뷰를 링크로 구성
  if [ ! -e DATA/mesh/topology.obj ] && [ -e DATA/static_300k/topology.obj ]; then
    echo "[setup] 파이프라인 레이아웃 감지 — DATA/mesh 를 심볼릭 링크로 구성합니다"
    mkdir -p DATA/mesh
    ln -sfn ../static_300k/topology.obj DATA/mesh/topology.obj
    ln -sfn ../static_300k/verts        DATA/mesh/verts
    if [ -e DATA/index/meta_all.csv ]; then
      ln -sfn ../index/meta_all.csv DATA/mesh/meta_all.csv
    elif [ -e DATA/static_300k/meta_all.csv ]; then
      ln -sfn ../static_300k/meta_all.csv DATA/mesh/meta_all.csv
    else
      echo "[ERR] meta_all.csv 가 없습니다 (DATA/index/, DATA/static_300k/ 모두) —" >&2
      echo "      파이프라인에서 qa 서비스(make_qa.sh 의 build_index)를 먼저 한 번 돌리세요." >&2
      return 1
    fi
    if [ -e DATA/outputs/pool_meta.csv ]; then
      ln -sfn ../outputs/pool_meta.csv DATA/mesh/pool_meta.csv
    else
      echo "[ERR] DATA/outputs/pool_meta.csv 가 없습니다 — make_mesh.sh 를 먼저 돌리세요." >&2
      return 1
    fi
  fi

  for p in DATA/mesh/topology.obj DATA/mesh/pool_meta.csv DATA/mesh/meta_all.csv; do
    [ -e "$p" ] || {
      echo "[ERR] 필요한 파일이 없습니다: $p" >&2
      echo "      DATA_DIR 를 파이프라인 DATA_DIR 로 지정하거나," >&2
      echo "      docs/data.md 의 매핑대로 DATA/mesh 를 직접 구성하세요." >&2
      return 1
    }
  done
}

ensure_split() {
  if [ ! -e DATA/mesh/train.txt ] || [ ! -e DATA/mesh/val.txt ]; then
    echo "[setup] split 파일이 없어 생성합니다 (--data DATA/mesh --out DATA/mesh)"
    python datasets/split_trainvaltest.py --data DATA/mesh --out DATA/mesh
  fi
}
