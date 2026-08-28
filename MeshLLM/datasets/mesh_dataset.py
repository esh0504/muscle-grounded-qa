"""Mesh dataset: 3D surface displacement → 11-D muscle activations.

Layout (DATA/mesh, produced from the pipeline outputs — docs/data.md):
    topology.obj                       shared rest pose + faces (370 surface verts)
    verts/shard_%05d.bin               big-endian float32 (n, 370, 3); the last
                                       shard may hold fewer than 1000 samples
    pool_meta.csv     per-index muscle activations + section meta
    meta_all.csv                       per-index simulation validity labels
    train.txt / val.txt / test.txt     splits (datasets/split_trainvaltest.py)

Input  : per-vertex displacement from topology.obj rest  (N_surf, 3) float32
Label  : muscle activations (11,) float32 in [0, 1]
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.split_trainvaltest import load_split_indices

MUSCLE_NAMES = [
    "GGP",
    "GGM",
    "GGA",
    "STY",
    "GH",
    "MH",
    "HG",
    "VERT",
    "TRANS",
    "IL",
    "SL",
]

N_SURF = 370
N_NODES = 948
SHARD_SIZE = 1000
DTYPE = ">f4"  # big-endian float32 as stored on disk

MUSCLE_CSV = "pool_meta.csv"
META_CSV = "meta_all.csv"
_EFFORT_RE = re.compile(r"@([0-9.]+)")


class MeshDataset(Dataset):
    """3D tongue surface displacement → 11 muscle activation labels.

    Prefer: ``MeshDataset(cfg.datasets)`` — config fields are read inside.
    """

    def __init__(self, cfg=None, **kwargs):
        super().__init__()
        p = self._as_dict(cfg)
        p.update(kwargs)

        root_dir = p.get("root_dir", None)
        split = p.get("split", "train")
        valid_only = bool(p.get("valid_only", True))
        geometry = p.get("geometry", "verts")
        split_dir = p.get("split_dir", None)

        self.root_dir = self._resolve_root(root_dir)
        self.split = split
        self.valid_only = valid_only
        self.geometry = geometry
        self.split_dir = Path(split_dir).resolve() if split_dir else self.root_dir
        if geometry not in ("verts", "nodes"):
            raise ValueError(f"geometry must be 'verts' or 'nodes', got {geometry!r}")

        info = self._read_topology_info()
        self.n_surf = int(info.get("n_surf_verts", N_SURF))
        self.n_nodes = int(info.get("n_fem_nodes", N_NODES))
        self.n_verts = self.n_surf if geometry == "verts" else self.n_nodes

        self.rest_verts, self.faces = self._load_topology()
        self.rest_path = self.root_dir / "topology.obj"
        if self.rest_verts.shape[0] != self.n_verts:
            raise ValueError(
                f"topology.obj has {self.rest_verts.shape[0]} verts, but geometry "
                f"'{self.geometry}' expects {self.n_verts}. Use geometry='verts'."
            )
        self.muscles = self._load_muscle_pool()
        self.muscle_meta = self._load_muscle_meta()

        catalog = self._index_samples()
        self.samples = self._select_split(catalog, split)
        self._mmap_cache: dict[tuple[str, int], np.ndarray] = {}

    @staticmethod
    def _as_dict(cfg: Any) -> dict:
        if cfg is None:
            return {}
        try:
            from omegaconf import OmegaConf

            if OmegaConf.is_config(cfg):
                return dict(OmegaConf.to_container(cfg, resolve=True))
        except Exception:
            pass
        if isinstance(cfg, Mapping):
            return dict(cfg)
        return {}

    # ------------------------------------------------------------------
    # paths / static assets
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_root(root_dir) -> Path:
        if root_dir is None:
            candidates = [Path("DATA/mesh"), Path("/workspace/DATA/mesh")]
        else:
            root = Path(root_dir)
            candidates = [root, root / "mesh"]
        for path in candidates:
            if (path / "topology.obj").is_file() and (path / MUSCLE_CSV).is_file():
                return path.resolve()
        raise FileNotFoundError(
            f"Could not find mesh data under root_dir={root_dir!r}. "
            f"Expected topology.obj and {MUSCLE_CSV}."
        )

    def _read_topology_info(self) -> dict:
        path = self.root_dir / "topology_info.txt"
        info = {}
        if not path.is_file():
            return info
        for line in path.read_text().splitlines():
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            info[key.strip()] = val.strip().split()[0]
        return info

    def _load_topology(self):
        """Load shared rest pose + faces from topology.obj."""
        path = self.root_dir / "topology.obj"
        if not path.is_file():
            raise FileNotFoundError(f"Missing topology.obj under {self.root_dir}")

        verts, faces = [], []
        for line in path.read_text().splitlines():
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([int(x.split("/")[0]) - 1 for x in line.split()[1:]])
        return (
            np.asarray(verts, dtype=np.float32),
            np.asarray(faces, dtype=np.int64),
        )

    # ------------------------------------------------------------------
    # muscle activations / sample meta — both from pool_meta.csv
    # ------------------------------------------------------------------
    def _muscle_rows(self):
        with (self.root_dir / MUSCLE_CSV).open() as fh:
            yield from csv.DictReader(fh)

    def _load_muscle_pool(self) -> np.ndarray:
        rows = list(self._muscle_rows())
        n = max(int(r["index"]) for r in rows) + 1
        acts = np.zeros((n, len(MUSCLE_NAMES)), dtype=np.float32)
        for row in rows:
            acts[int(row["index"])] = [float(row[m]) for m in MUSCLE_NAMES]
        return acts

    def _load_muscle_meta(self) -> dict[int, dict]:
        out: dict[int, dict] = {}
        for row in self._muscle_rows():
            detail = row.get("detail", "") or ""
            m = _EFFORT_RE.search(detail)
            out[int(row["index"])] = {
                # stratified-split key
                "block": row.get("section", "UNKNOWN"),
                "effort": float(m.group(1)) if m else 0.0,
                "n_active": int(row.get("n_active", 0) or 0),
                "phoneme": row.get("phoneme", ""),
                "detail": detail,
            }
        return out

    # ------------------------------------------------------------------
    # sample index — meta_all.csv is a single file; index is contiguous 0..N-1,
    # so shard/local follow directly from the index.
    # ------------------------------------------------------------------
    def _complete_shards(self) -> set[int]:
        geo_dir = self.root_dir / self.geometry
        done = set()
        for path in geo_dir.glob("shard_*.bin"):
            if path.name.endswith(".part"):
                continue
            done.add(int(path.stem.split("_")[1]))
        return done

    def _index_samples(self) -> dict[int, dict]:
        complete = self._complete_shards()
        catalog: dict[int, dict] = {}
        with (self.root_dir / META_CSV).open() as fh:
            for row in csv.DictReader(fh):
                label = row["label"]
                if self.valid_only and label != "VALID":
                    continue
                gidx = int(row["index"])
                if gidx >= len(self.muscles):
                    continue
                shard, local = divmod(gidx, SHARD_SIZE)
                if shard not in complete:
                    continue
                mm = self.muscle_meta.get(gidx, {})
                catalog[gidx] = {
                    "global_index": gidx,
                    "shard": shard,
                    "local": local,
                    "sim_label": label,
                    "block": mm.get("block", "UNKNOWN"),
                }
        if not catalog:
            raise RuntimeError(f"No usable samples found under {self.root_dir}")
        return catalog

    def _select_split(self, catalog: dict[int, dict], split: str) -> list[dict]:
        if split == "all":
            return [catalog[i] for i in sorted(catalog)]
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train|val|test|all, got {split!r}")

        indices = load_split_indices(self.split_dir, split)
        missing = [i for i in indices if i not in catalog]
        if missing:
            raise RuntimeError(
                f"{len(missing)} indices from {self.split_dir / (split + '.txt')} "
                f"are missing in the geometry catalog (e.g. {missing[:5]}). "
                "Regenerate splits with the same --geometry / valid_only settings."
            )
        return [catalog[i] for i in indices]

    # ------------------------------------------------------------------
    # geometry IO — the last shard holds fewer than SHARD_SIZE samples,
    # so reshape with -1 rather than a fixed size.
    # ------------------------------------------------------------------
    def _shard_array(self, shard: int) -> np.ndarray:
        key = (self.geometry, shard)
        arr = self._mmap_cache.get(key)
        if arr is not None:
            return arr
        path = self.root_dir / self.geometry / f"shard_{shard:05d}.bin"
        mm = np.memmap(path, dtype=DTYPE, mode="r")
        arr = np.asarray(mm).reshape(-1, self.n_verts, 3)
        self._mmap_cache[key] = arr
        return arr

    def _load_geometry(self, shard: int, local: int) -> np.ndarray:
        return np.array(self._shard_array(shard)[local], dtype=np.float32, order="C")

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        rec = self.samples[index]
        gidx = rec["global_index"]
        verts = self._load_geometry(rec["shard"], rec["local"])
        # Per-vertex displacement from rest pose (not absolute coordinates).
        disp = verts - self.rest_verts
        muscle = self.muscles[gidx].copy()
        mm = self.muscle_meta.get(gidx, {})

        return {
            "inputs": torch.from_numpy(disp),  # (V, 3) displacement
            "label": torch.from_numpy(muscle),  # (11,)
            "index": gidx,
            "faces": torch.from_numpy(self.faces),  # (F, 3)
            "rest": torch.from_numpy(self.rest_verts),  # (V, 3) shared rest
            "block": mm.get("block", ""),
            "effort": float(mm.get("effort", 0.0)),
            "n_active": int(mm.get("n_active", 0)),
            "sim_label": rec["sim_label"],
            "muscle_names": MUSCLE_NAMES,
        }

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        return {
            "inputs": torch.stack([b["inputs"] for b in batch], dim=0),
            "label": torch.stack([b["label"] for b in batch], dim=0),
            "index": torch.tensor([b["index"] for b in batch], dtype=torch.long),
            "faces": batch[0]["faces"],
            "rest": batch[0]["rest"],
            "block": [b["block"] for b in batch],
            "effort": torch.tensor([b["effort"] for b in batch], dtype=torch.float32),
            "n_active": torch.tensor([b["n_active"] for b in batch], dtype=torch.long),
            "sim_label": [b["sim_label"] for b in batch],
            "muscle_names": batch[0]["muscle_names"],
        }
