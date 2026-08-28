# -*- coding: utf-8 -*-
"""config.json loader. All pipeline scripts resolve paths through here.

Usage:
    from modules.config import CFG, path
    pool = path("pool_txt")          # absolute path
    n    = CFG["pool_total"]

Override the config file with the PIPELINE_CONFIG environment variable.
Relative paths in config.json are resolved against the config file's directory.
"""
import json
import os

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CFG_FILE = os.environ.get(
    "PIPELINE_CONFIG", os.path.join(_PKG_ROOT, "settings", "config.json"))
# Relative paths in the config resolve against the PIPELINE ROOT (not settings/)
_ROOT = (_PKG_ROOT if os.path.dirname(os.path.abspath(_CFG_FILE))
         == os.path.join(_PKG_ROOT, "settings")
         else os.path.dirname(os.path.abspath(_CFG_FILE)))

with open(_CFG_FILE, encoding="utf-8") as _f:
    CFG = json.load(_f)


def path(key):
    """Return CFG[key] as an absolute path (relative paths anchor at the config dir)."""
    p = CFG[key]
    if not os.path.isabs(p):
        p = os.path.join(_ROOT, p)
    return os.path.normpath(p)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


# Pipeline root directory (parent of modules/) — for locating settings/ and assets/
PIPELINE_ROOT = _PKG_ROOT
