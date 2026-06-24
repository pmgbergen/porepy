"""Plot script for case 2.

Used to create plots for publication below.

"""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# Fetching data stored in directories.
c2a_dir: str = "visualization/CI_CASE2A/"

c2a_sdirs = [p.name for p in Path(c2a_dir).iterdir() if p.is_dir()]

print("Found Case 2a dirs:\n" + "\n".join(c2a_sdirs))


# Read data
def fetch(folder: str, sim: str) -> dict:
    p = f"{folder}{sim}"
    path = Path(p + "/solver_statistics.json").resolve()
    if not path.is_file():
        raise ValueError(f"Statistics file not found for {p}")

    return json.load(path.open("r"))


c2a_data = [
    fetch(c2a_dir, p)
    for p in c2a_sdirs
    if ("EPRIM_False" not in p) and ("ICHOR_False" not in p)
]
c2a_e = [fetch(c2a_dir, p) for p in c2a_sdirs if "EPRIM_False" in p][0]
c2a_npc = [fetch(c2a_dir, p) for p in c2a_sdirs if "ICHOR_False" in p][0]
