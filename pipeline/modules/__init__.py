# -*- coding: utf-8 -*-
"""modules — shared library for the 3DTongueQA construction pipeline.

Modules
  config    : config.json loader (paths / constants)
  muscles   : 11-muscle inventory + ground-truth anatomical functions (KO/EN)
  anchors   : literature-informed vowel centers + consonant rules (Step 1 spec)
  dorsum    : fixed mid-sagittal dorsum vertex indices + contour reader
  features  : 32 palate-normalized articulatory features
  mesh_io   : shard binary readers (verts/nodes) + meta loaders
  spans     : mask_spans builders (muscle/number/movement/region tagging)
"""
