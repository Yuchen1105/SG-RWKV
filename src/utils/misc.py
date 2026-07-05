# src/utils/misc.py
from __future__ import annotations

import os
import copy
from typing import Iterable

def list_image_files(folder: str, exts: Iterable[str]) -> list[str]:
    exts = tuple(e.lower() for e in exts)
    out = []
    for fn in os.listdir(folder):
        if fn.lower().endswith(exts):
            out.append(os.path.join(folder, fn))
    return sorted(out)

def stem(path: str) -> str:
    # basename without extension
    base = os.path.basename(path)
    return os.path.splitext(base)[0]
