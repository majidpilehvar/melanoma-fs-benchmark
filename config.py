"""Paths, resolved once for every script.

Data location, in order of precedence:
  1. environment variable MELANOMA_DATA
  2. ./data/Data.csv next to these scripts
  3. ./Main/Data.csv (layout of the original thesis project)
Results and figures are written next to the scripts unless MELANOMA_OUT is set.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_data():
    env = os.environ.get("MELANOMA_DATA")
    if env:
        return env
    for cand in (os.path.join(HERE, "data", "Data.csv"),
                 os.path.join(HERE, "Main", "Data.csv"),
                 os.path.join(HERE, "Data.csv"),
                 "/mnt/user-data/uploads/Data.csv"):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        "Data.csv not found. Put it in ./data/ or set MELANOMA_DATA=/path/to/Data.csv")


DATA = _find_data()
OUT_ROOT = os.environ.get("MELANOMA_OUT", HERE)
RESULTS = os.path.join(OUT_ROOT, "results")
FIGURES = os.path.join(OUT_ROOT, "figures")
CKPT = os.path.join(RESULTS, "ckpt")
for _d in (RESULTS, FIGURES, CKPT):
    os.makedirs(_d, exist_ok=True)
