#!/usr/bin/env python3
"""Create per-section stratified train/val/test index lists.

Writes one global sample index per line:
  <out_dir>/train.txt
  <out_dir>/val.txt
  <out_dir>/test.txt

Default split ratio is 9 : 0.5 : 0.5 within each section
(rest / single / pair / triple / anchor / effort / neighbor / spacefill),
so Stage 1 and Stage 2 share the same val/test mesh boundaries.

Usage:
  python datasets/split_trainvaltest.py --data DATA/mesh --out DATA/mesh
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SHARD_SIZE = 1000
DEFAULT_RATIO = (9.0, 0.5, 0.5)

MUSCLE_CSV = "pool_meta.csv"
META_CSV = "meta_all.csv"


def _is_data_dir(path: Path) -> bool:
    return (path / META_CSV).is_file() and (path / MUSCLE_CSV).is_file()


def resolve_data_dir(data: Path | None) -> Path:
    candidates = []
    if data is not None:
        candidates += [data, data / "mesh"]
    candidates += [Path("DATA/mesh"), Path("/workspace/DATA/mesh")]
    for path in candidates:
        if _is_data_dir(path):
            return path.resolve()
    raise FileNotFoundError(
        f"데이터 폴더를 못 찾았습니다 (tried {[str(c) for c in candidates]}); "
        f"{META_CSV} + {MUSCLE_CSV} 가 있어야 합니다"
    )


def complete_shards(data_dir: Path, geometry: str = "verts") -> set[int]:
    done = set()
    for path in (data_dir / geometry).glob("shard_*.bin"):
        if path.name.endswith(".part"):
            continue
        done.add(int(path.stem.split("_")[1]))
    return done


def load_block_map(data_dir: Path) -> dict[int, str]:
    """층화 키: 근육 csv 의 'section' 열."""
    out = {}
    with (data_dir / MUSCLE_CSV).open() as fh:
        for row in csv.DictReader(fh):
            out[int(row["index"])] = row["section"]
    return out


def collect_indices(
    data_dir: Path,
    *,
    valid_only: bool = True,
    geometry: str = "verts",
) -> list[tuple[int, str]]:
    """Return [(global_index, section), ...] for usable simulated samples."""
    blocks = load_block_map(data_dir)
    complete = complete_shards(data_dir, geometry=geometry)
    pairs: list[tuple[int, str]] = []

    # meta_all.csv 한 파일. index 가 0..N-1 연속이라 shard 는 index//SHARD_SIZE.
    with (data_dir / META_CSV).open() as fh:
        for row in csv.DictReader(fh):
            if valid_only and row["label"] != "VALID":
                continue
            gidx = int(row["index"])
            if gidx not in blocks or gidx // SHARD_SIZE not in complete:
                continue
            pairs.append((gidx, blocks[gidx]))
    pairs.sort(key=lambda x: x[0])
    return pairs


def normalize_ratio(ratio) -> tuple[float, float, float]:
    w = np.asarray(ratio, dtype=np.float64)
    if w.shape != (3,) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError(f"ratio must be 3 non-negative weights, got {ratio!r}")
    w = w / w.sum()
    return float(w[0]), float(w[1]), float(w[2])


def stratified_split(
    pairs: list[tuple[int, str]],
    ratio=(9.0, 0.5, 0.5),
    seed: int = 42,
) -> dict[str, list[int]]:
    """Per-section train/val/test split. Tiny sections stay in train."""
    r_train, r_val, r_test = normalize_ratio(ratio)
    rng = np.random.default_rng(seed)

    by_block: dict[str, list[int]] = defaultdict(list)
    for gidx, block in pairs:
        by_block[block].append(gidx)

    splits = {"train": [], "val": [], "test": []}
    for block in sorted(by_block):
        ids = np.asarray(by_block[block], dtype=np.int64)
        n = len(ids)
        order = rng.permutation(n)

        n_val = int(n * r_val)
        n_test = int(n * r_test)
        if n_val + n_test >= n:
            n_val = min(n_val, max(0, n - 1))
            n_test = min(n_test, max(0, n - 1 - n_val))
        n_train = n - n_val - n_test

        splits["train"].extend(ids[order[:n_train]].tolist())
        splits["val"].extend(ids[order[n_train : n_train + n_val]].tolist())
        splits["test"].extend(ids[order[n_train + n_val :]].tolist())

    for key in splits:
        splits[key] = sorted(int(x) for x in splits[key])
    return splits


def write_split_files(splits: dict[str, list[int]], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, indices in splits.items():
        path = out_dir / f"{name}.txt"
        path.write_text("".join(f"{i}\n" for i in indices))
        paths[name] = path
    return paths


def load_split_indices(split_dir: Path, split: str) -> list[int]:
    """Read global indices from train.txt / val.txt / test.txt."""
    path = Path(split_dir) / f"{split}.txt"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run: python datasets/split_trainvaltest.py --out {split_dir}"
        )
    indices = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        indices.append(int(line.split()[0]))
    return indices


def main():
    ap = argparse.ArgumentParser(description="Write train/val/test index lists")
    ap.add_argument("--data", type=Path, default=None, help="DATA/mesh root")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Directory for train.txt/val.txt/test.txt (default: <data>)",
    )
    ap.add_argument("--ratio", type=float, nargs=3, default=list(DEFAULT_RATIO), metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--geometry", choices=("verts", "nodes"), default="verts")
    ap.add_argument("--include-invalid", action="store_true", help="Also keep non-VALID rows")
    args = ap.parse_args()

    data_dir = resolve_data_dir(args.data)
    out_dir = (args.out or data_dir).resolve()

    pairs = collect_indices(
        data_dir,
        valid_only=not args.include_invalid,
        geometry=args.geometry,
    )
    splits = stratified_split(pairs, ratio=args.ratio, seed=args.seed)
    paths = write_split_files(splits, out_dir)

    # Summary
    block_all = Counter(b for _, b in pairs)
    print(f"data: {data_dir}")
    print(f"out:  {out_dir}")
    print(f"ratio {args.ratio[0]}:{args.ratio[1]}:{args.ratio[2]}  seed={args.seed}")
    print(
        f"total {len(pairs)} → "
        f"train={len(splits['train'])}  val={len(splits['val'])}  test={len(splits['test'])}"
    )

    # Per-section counts from written files
    idx_to_block = dict(pairs)
    print(f"{'section':10s} {'all':>7s} {'train':>7s} {'val':>7s} {'test':>7s}")
    for block in sorted(block_all):
        counts = {
            name: sum(1 for i in splits[name] if idx_to_block[i] == block)
            for name in ("train", "val", "test")
        }
        print(
            f"{block:10s} {block_all[block]:7d} "
            f"{counts['train']:7d} {counts['val']:7d} {counts['test']:7d}"
        )

    for name, path in paths.items():
        print(f"wrote {path} ({len(splits[name])} indices)")


if __name__ == "__main__":
    main()
