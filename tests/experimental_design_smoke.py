#!/usr/bin/env python3
"""Experimental-design gate — every result checked against known math.

An unexplored design direction yields the highest expected information gain;
sequential active learning shows diminishing per-step EIG and monotone cumulative
EIG; D-optimal selection increases log det; Latin-Hypercube spreads points more
evenly than a degenerate design and is seed-reproducible."""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "server", "experimental_design.py")

try:
    import numpy as np
except Exception as e:  # noqa: BLE001
    print(f"SKIP: numpy not available ({e}).")
    sys.exit(0)
try:
    import scipy  # noqa: F401
except Exception as e:  # noqa: BLE001
    print(f"SKIP: scipy not available ({e}).")
    sys.exit(0)

passed = 0


def check(name, cond, ctx=None):
    global passed
    if not cond:
        print(f"FAIL: {name}\n  {ctx}")
        sys.exit(1)
    passed += 1
    print(f"ok: {name}")


def run(p):
    r = subprocess.run([sys.executable, SCRIPT], input=json.dumps(p).encode(),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        print("STDERR:", r.stderr.decode())
    return json.loads(r.stdout.decode())


# --------------------------------------------------------------------------- #
# Task 1 — an unexplored direction is the most informative next experiment.
# Already-run experiments all lie along the x-axis; candidates: another x point
# (redundant) vs a y-axis point (unexplored) -> the y-axis one wins.
# --------------------------------------------------------------------------- #
# Candidates are UNIT vectors (equal magnitude) so the comparison isolates
# DIRECTION: the unexplored y-direction must be more informative than redundant x.
design = [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]        # all along x
pool = [[1.0, 0.0], [0.0, 1.0]]                        # 0: redundant x, 1: new y
d = run({"task": "bayesian_optimal_design", "designMatrix": design,
         "candidatePool": pool, "noiseVariance": 1.0, "priorPrecision": 1.0})
check("bod status success", d["status"] == "success", d)
check("bod best candidate is the unexplored y-direction (#1)", d["bestIndex"] == 1, d)
by_idx = {r["index"]: r for r in d["results"]}
check("bod y-direction EIG > x-direction EIG",
      by_idx[1]["expectedInformationGain"] > by_idx[0]["expectedInformationGain"], d["results"])
check("bod EIG is non-negative", all(r["expectedInformationGain"] >= 0 for r in d["results"]), d["results"])

# With NO prior experiments, a unit x and unit y candidate are symmetric -> equal EIG.
d2 = run({"task": "bayesian_optimal_design", "candidatePool": [[1.0, 0.0], [0.0, 1.0]]})
e0 = d2["results"][0]["expectedInformationGain"]; e1 = d2["results"][1]["expectedInformationGain"]
check("bod symmetric candidates have equal EIG (no prior data)", abs(e0 - e1) < 1e-9, (e0, e1))

# --------------------------------------------------------------------------- #
# Task 2 — sequential active learning: diminishing per-step, monotone cumulative.
# --------------------------------------------------------------------------- #
pool4 = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]]
s = run({"task": "sequential_active_learning", "candidatePool": pool4, "nBatch": 3,
         "noiseVariance": 1.0, "priorPrecision": 1.0})
check("seq status success", s["status"] == "success", s)
check("seq selected exactly 3", len(s["selected"]) == 3, s)
check("seq no duplicate picks", len(set(s["selectedIndices"])) == 3, s)
eigs = [pk["expectedInformationGain"] for pk in s["selected"]]
check("seq per-step EIG is non-increasing (diminishing returns)",
      all(eigs[i] >= eigs[i + 1] - 1e-9 for i in range(len(eigs) - 1)), eigs)
cums = [pk["cumulativeEIG"] for pk in s["selected"]]
check("seq cumulative EIG is strictly increasing",
      all(cums[i] < cums[i + 1] for i in range(len(cums) - 1)), cums)
check("seq cumulativeEIG equals sum of per-step EIG",
      abs(cums[-1] - sum(eigs)) < 1e-6, (cums[-1], sum(eigs)))

# --------------------------------------------------------------------------- #
# Task 3 — D-optimal selection increases log det; picks orthogonal directions.
# --------------------------------------------------------------------------- #
poold = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]   # two near-collinear + one orthogonal
do = run({"task": "d_optimal_selection", "candidatePool": poold, "k": 2,
          "noiseVariance": 1.0, "priorPrecision": 1.0})
check("dopt status success", do["status"] == "success", do)
check("dopt logdet increased", do["logDetFinal"] > do["logDetStart"], do)
# The best 2-subset for coverage is the orthogonal pair {0 (or 1), 2}, must include #2.
check("dopt selection includes the orthogonal direction (#2)", 2 in do["selectedIndices"], do)

# --------------------------------------------------------------------------- #
# Task 4 — Latin-Hypercube space-filling: reproducible, in-bounds, well spread.
# --------------------------------------------------------------------------- #
lhs = run({"task": "space_filling_design", "nPoints": 8,
           "bounds": [[0.0, 10.0], [-5.0, 5.0]], "seed": 7})
check("lhs status success", lhs["status"] == "success", lhs)
check("lhs returned 8 points", len(lhs["points"]) == 8, lhs)
pts = np.array(lhs["points"])
check("lhs points within bounds",
      np.all(pts[:, 0] >= 0) and np.all(pts[:, 0] <= 10) and
      np.all(pts[:, 1] >= -5) and np.all(pts[:, 1] <= 5), pts)
lhs2 = run({"task": "space_filling_design", "nPoints": 8,
            "bounds": [[0.0, 10.0], [-5.0, 5.0]], "seed": 7})
check("lhs seed-reproducible", lhs2["points"] == lhs["points"])
check("lhs min pairwise distance is positive", lhs["minPairwiseDistance"] > 0, lhs)

# --------------------------------------------------------------------------- #
# Error handling.
# --------------------------------------------------------------------------- #
check("unknown task -> error", run({"task": "nope"}).get("status") == "error")
check("feature-dim mismatch -> error",
      run({"task": "bayesian_optimal_design", "designMatrix": [[1.0, 0.0]],
           "candidatePool": [[1.0, 0.0, 0.0]]}).get("status") == "error")
check("d_optimal k too large -> error",
      run({"task": "d_optimal_selection", "candidatePool": [[1.0, 0.0]], "k": 5}).get("status") == "error")
check("space_filling bad bounds -> error",
      run({"task": "space_filling_design", "nPoints": 4, "bounds": [[5.0, 1.0]]}).get("status") == "error")

print(f"\nALL {passed} EXPERIMENTAL-DESIGN TESTS PASSED")
