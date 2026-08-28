#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V2/modules/utils.py

프로젝트 공통 유틸 — 파일 IO와 변환 유틸 ).
artisynth·retarget 알고리즘 어디에도 종속되지 않는다(모델은 duck typing:
model.verts / model.faces / model.names 만 사용). 알고리즘 모듈은 여기의
프리미티브(load/save/vis)를 호출해서 쓰고, 자기 자신은 계산에만 집중한다.

구성:
  · 경로        V2_DIR / REPO_DIR / OUT_DIR / ensure_dir / out_path / repo_path / data_path
  · 이미지 IO   save_png   (save_gif 는 미사용이라 dummy/unused_utils.py 로 옮겼다)
  · OBJ  IO     load_obj / extract_obj / save_obj
  · mask IO     load_mask / load_video / mask_label_2d
  · CSV / npy   save_csv / read_csv_dicts / save_npy
  · JSONL IO    write_jsonl
  · 변환/설정   move (배치→device) / load_experiment_cfg (Hydra 실험 설정)
  · 시각화      visualize.py 로 옮겼다 — visualization (+ vis / vis3d /
                vis_mask / vis_with_activations 호환 래퍼). 하위호환은 이 모듈
                하단 ``__getattr__`` 이 지연 재수출로 처리한다.

알고리즘(dummy/artisynth/·retarget/)은 여기의 프리미티브를 호출해서 IO/렌더를 처리한다.
"""
import csv
import glob
import json
import os
import re
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# 경로
# --------------------------------------------------------------------------- #
V2_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.path.dirname(V2_DIR)
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(V2_DIR, "_test_out"))


def ensure_dir(directory):
    """디렉터리 생성(존재하면 무시). 인자를 그대로 반환."""
    if directory:
        os.makedirs(directory, exist_ok=True)
    return directory


def out_path(name):
    """OUT_DIR 아래 경로를 만들고 OUT_DIR을 보장한다. name이 절대경로면 그대로."""
    ensure_dir(OUT_DIR)
    return name if os.path.isabs(name) else os.path.join(OUT_DIR, name)


def repo_path(*parts):
    """리포 루트(=V2의 부모) 기준 경로.

    예: repo_path("tongue_model", "tongue_rest_m.obj")
        repo_path("datasets", "GT_Segmentations", "Subject3")
    """
    return os.path.join(REPO_DIR, *parts)


def data_path(data_root, *parts):
    """data_root 기준 경로. data_root가 상대면 V2_DIR(프로젝트 루트) 기준."""
    root = data_root if os.path.isabs(data_root) else os.path.join(V2_DIR, data_root)
    return os.path.normpath(os.path.join(root, *parts))


# --------------------------------------------------------------------------- #
# 이미지 IO
# --------------------------------------------------------------------------- #
def save_png(img, name_or_path):
    """(H, W, 3) uint8 → PNG 저장. 반환: 저장 경로(str) 또는 None.

    bare name(디렉터리 구분자 없음)이면 OUT_DIR 아래에 저장한다.
    imageio가 없으면 조용히 건너뛰고 None을 반환한다.
    """
    if os.path.isabs(name_or_path) or os.path.dirname(name_or_path):
        path = name_or_path
    else:
        path = out_path(name_or_path)
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    try:
        import imageio.v2 as imageio
        imageio.imwrite(path, img)
        return path
    except Exception as e:
        print("   (PNG 저장 건너뜀: %s)" % e)
        return None


def save_npy(path, arr):
    """np.save 래퍼(상위 디렉터리 보장). 저장 경로 반환."""
    ensure_dir(os.path.dirname(os.path.abspath(path)) or ".")
    np.save(path, np.asarray(arr))
    return path


# --------------------------------------------------------------------------- #
# CSV IO (프리미티브)
# --------------------------------------------------------------------------- #
def save_csv(path, fieldnames, rows):
    """rows(list[dict])를 fieldnames 순서로 CSV 저장. 저장 경로 반환."""
    out = os.path.abspath(str(path))
    ensure_dir(os.path.dirname(out) or ".")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return out


def read_csv_dicts(path):
    """CSV → list[dict] (csv.DictReader)."""
    p = os.path.abspath(str(path))
    if not os.path.isfile(p):
        raise FileNotFoundError("csv not found: %s" % p)
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------- #
# OBJ IO
# --------------------------------------------------------------------------- #
DEFAULT_MESH_COLOR = (230, 90, 75)


def _require_mesh(model):
    """model.verts / model.faces 존재 검증(duck typing)."""
    if (model is None or getattr(model, "verts", None) is None
            or getattr(model, "faces", None) is None):
        raise ValueError("mesh가 비었습니다 (verts/faces 필요).")


def load_obj(path):
    """Wavefront OBJ → (verts (N,3) float, faces (F,3) int).

    다각형 face는 fan 삼각분할. 단위는 파일 그대로(혀 OBJ는 metres)."""
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            t = line.split()
            if not t:
                continue
            if t[0] == "v":
                verts.append([float(t[1]), float(t[2]), float(t[3])])
            elif t[0] == "f":
                idx = [int(p.split("/")[0]) - 1 for p in t[1:]]
                for k in range(1, len(idx) - 1):       # fan triangulation
                    faces.append([idx[0], idx[k], idx[k + 1]])
    return np.asarray(verts, dtype=float), np.asarray(faces, dtype=int)


def extract_obj(model, color=None):
    """TongueModel(또는 verts/faces 핸들) → OBJ 데이터 dict.

    반환 키:
      points_cloud : (N, 3) float — 정점 좌표 (metres)
      Mesh         : (F, 3) int   — 삼각형 face 인덱스 (0-based)
      Color        : (N, 3) uint8 — 정점 RGB (0..255)
    """
    _require_mesh(model)
    verts = np.asarray(model.verts, dtype=float)
    faces = np.asarray(model.faces, dtype=int)
    n = len(verts)
    if color is None:
        rgb = np.tile(np.asarray(DEFAULT_MESH_COLOR, dtype=np.uint8), (n, 1))
    else:
        c = np.asarray(color, dtype=np.uint8)
        rgb = np.tile(c.reshape(1, 3), (n, 1)) if c.shape == (3,) else c.reshape(n, 3)
    return {"points_cloud": verts, "Mesh": faces, "Color": rgb}


def _obj_path(path):
    p = str(path)
    return p if p.lower().endswith(".obj") else p + ".obj"


def save_obj(obj, path):
    """extract_obj() 결과(동일 키 dict)를 Wavefront OBJ로 저장. 저장 경로 반환.

    Color가 있으면 ``v x y z r g b`` (RGB 0..1) 확장 형식으로 쓴다.
    """
    verts = np.asarray(obj["points_cloud"], dtype=float)
    faces = obj.get("Mesh")
    colors = obj.get("Color")
    if colors is not None:
        colors = np.asarray(colors, dtype=np.uint8).reshape(len(verts), 3)

    out = _obj_path(path)
    ensure_dir(os.path.dirname(os.path.abspath(out)) or ".")
    with open(out, "w", encoding="utf-8") as f:
        f.write("# modules.utils save_obj\n")
        for i, v in enumerate(verts):
            if colors is not None:
                r, g, b = colors[i] / 255.0
                f.write("v %.6f %.6f %.6f %.6f %.6f %.6f\n"
                        % (v[0], v[1], v[2], r, g, b))
            else:
                f.write("v %.6f %.6f %.6f\n" % (v[0], v[1], v[2]))
        if faces is not None:
            for t in np.asarray(faces, dtype=int):
                f.write("f %d %d %d\n" % (t[0] + 1, t[1] + 1, t[2] + 1))
    return out


# --------------------------------------------------------------------------- #
# MRI mask IO
# --------------------------------------------------------------------------- #
MAT_VAR_NAME = "mask_frame"


def _natkey(path):
    nums = re.findall(r"\d+", os.path.basename(path))
    return int(nums[-1]) if nums else 0


def _require_dir(folder_path, label="folder_path"):
    folder = os.path.abspath(str(folder_path))
    if not os.path.isdir(folder):
        raise NotADirectoryError("%s is not a directory: %s" % (label, folder))
    return folder


def mask_label_2d(mask):
    """(H,W,C) 또는 (H,W) → 2D label slice."""
    mask = np.asarray(mask)
    if mask.ndim == 3:
        return mask[..., 0]
    if mask.ndim == 2:
        return mask
    raise ValueError("mask must be (H,W) or (H,W,C), got %s" % (mask.shape,))


def _as_hwc(arr):
    """array → (H, W, C). 2D label/image → C=1."""
    a = np.asarray(arr)
    if a.ndim == 2:
        return a[..., np.newaxis]
    if a.ndim == 3:
        return a
    raise ValueError("mask must be 2D (H,W) or 3D (H,W,C), got %s" % (a.shape,))


def _load_mat(path, mat_var=MAT_VAR_NAME):
    import scipy.io as sio
    data = sio.loadmat(path)
    if mat_var in data:
        return data[mat_var]
    for key, val in data.items():
        if not key.startswith("__"):
            return val
    raise ValueError("no array found in %s" % path)


def load_mask(mask_path, mat_var=MAT_VAR_NAME):
    """단일 마스크 파일(.mat/.npy/.npz/이미지) → (H, W, C)."""
    path = os.path.abspath(str(mask_path))
    if not os.path.isfile(path):
        raise FileNotFoundError("mask_path not found: %s" % path)

    ext = os.path.splitext(path)[1].lower()
    if ext == ".mat":
        arr = _load_mat(path, mat_var=mat_var)
    elif ext == ".npy":
        arr = np.load(path)
    elif ext == ".npz":
        data = np.load(path)
        if mat_var in data:
            arr = data[mat_var]
        else:
            keys = [k for k in data.files if not k.startswith("_")]
            if not keys:
                raise ValueError("empty npz: %s" % path)
            arr = data[keys[0]]
    elif ext in (".png", ".tif", ".tiff", ".bmp"):
        try:
            import imageio.v2 as imageio
            arr = imageio.imread(path)
        except Exception:
            from PIL import Image
            arr = np.asarray(Image.open(path))
    else:
        raise ValueError("unsupported mask format: %s" % ext)
    return _as_hwc(arr)


def load_video(folder_path):
    """폴더 내 ``mask_*.mat`` → (T, H, W, C)."""
    folder = _require_dir(folder_path)
    files = sorted(glob.glob(os.path.join(folder, "mask_*.mat")), key=_natkey)
    if not files:
        raise FileNotFoundError("no mask_*.mat files under %s" % folder)
    frames = [load_mask(p) for p in files]
    return np.stack(frames, axis=0)


# --------------------------------------------------------------------------- #
# JSONL IO
# --------------------------------------------------------------------------- #
def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fo:
        for r in rows:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# 텐서 배치 이동
# --------------------------------------------------------------------------- #
def move(batch, device):
    # torch 는 함수 안에서 import 한다 — utils 는 dummy/artisynth/·test.py 처럼 torch 가
    # 필요 없는 쪽에서도 import 되므로 모듈 레벨 의존성을 늘리지 않는다.
    import torch

    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


# --------------------------------------------------------------------------- #
# Hydra 실험 설정 로더
# --------------------------------------------------------------------------- #
def load_experiment_cfg(name, overrides=None):
    """``configs/experiment/<name>.yaml`` 오버레이를 얹은 train_s2 설정을 만든다.

    train_s2.py 의 argparse ``apply_experiment_cfg(ap, args, argv=[])`` 를 대체한다.
    dummy/scripts/gates.py, dummy/scripts/gate4_alignment.py 가 공유한다.

    hydra 는 **함수 안에서 지연 import** 한다 — utils.py 는 hydra 없이도 import 돼야
    한다(dummy/artisynth/ 등이 여기의 IO 프리미티브만 쓴다).
    """
    from hydra import compose, initialize_config_dir

    ov = ["+experiment=%s" % name]
    if overrides:
        ov.extend(list(overrides))
    with initialize_config_dir(config_dir=str(Path.cwd() / "configs"),
                               version_base=None):
        return compose(config_name="train_s2", overrides=ov)
