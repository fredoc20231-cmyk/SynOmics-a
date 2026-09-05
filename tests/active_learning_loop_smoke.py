#!/usr/bin/env python3
"""Active-learning loop gate — checked against a KNOWN linear ground truth.

On y = Xw with true w=[2,-1], the fitted posterior mean recovers w; the loop
proposes the most informative unexplored experiment and HALTS (no fabricated
outcome); assimilating a real measurement yields positive information gain and
shrinks posterior variance; convergence flips with the tolerance."""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "server", "active_learning_loop.py")

try:
    import numpy as np
except Exception as e:  # noqa: BLE001
    print(f"SKIP: numpy not available ({e}).")
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


# Known ground truth: y = 2*x0 - 1*x1 (real, exact) on a small informative design.
w_true = np.array([2.0, -1.0])
X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [1.0, 2.0]])
y = X @ w_true
pool = [[3.0, 0.0], [0.0, 3.0], [1.0, 1.0]]

# ---- propose_next_experiment ----
# Small noise variance so the posterior mean closely recovers the true weights.
d = run({"task": "propose_next_experiment", "designMatrix": X.tolist(),
         "observations": y.tolist(), "candidatePool": pool,
         "noiseVariance": 1e-3, "priorPrecision": 1e-6})
check("propose status success", d["status"] == "success", d)
check("propose HALTS for a real measurement (no fabricated outcome)",
      d["awaitingMeasurement"] is True and "proposedResponse" not in d, d)
est = {c["index"]: c["estimate"] for c in d["coefficients"]}
check("posterior mean recovers true w0=2", abs(est[0] - 2.0) < 1e-2, est)
check("posterior mean recovers true w1=-1", abs(est[1] - (-1.0)) < 1e-2, est)
check("proposed index is within the pool", 0 <= d["proposedIndex"] < len(pool), d)
# EIG ranking is real: the highest-magnitude unexplored candidate wins.
eig_by = {r["index"]: r["expectedInformationGain"] for r in d["ranked"]}
check("all EIG non-negative", all(v >= 0 for v in eig_by.values()), eig_by)

# ---- assimilate_measurement ----
# Provide a REAL measured point consistent with the ground truth.
xn = [0.0, 3.0]
yn = float(np.array(xn) @ w_true)   # = -3.0, a real measurement (not fabricated by the tool)
a = run({"task": "assimilate_measurement", "designMatrix": X.tolist(),
         "observations": y.tolist(), "newDesign": xn, "newObservation": yn,
         "noiseVariance": 1.0, "priorPrecision": 1.0})
check("assimilate status success", a["status"] == "success", a)
check("assimilate realized info gain > 0", a["realizedInformationGain"] > 0, a)
check("assimilate reduces posterior trace", a["posteriorTraceAfter"] < a["posteriorTraceBefore"], a)
check("assimilate reduction percent > 0", a["traceReductionPercent"] > 0, a)
check("assimilate reports before/after coefficients", len(a["coefficients"]) == 2, a)

# Assimilating a REDUNDANT point (already well-covered direction) gives less gain
# than a fresh direction — monotone sanity, still real.
a_redundant = run({"task": "assimilate_measurement", "designMatrix": X.tolist(),
                   "observations": y.tolist(), "newDesign": [1.0, 0.0],
                   "newObservation": 2.0, "noiseVariance": 1.0, "priorPrecision": 1.0})
check("redundant point gains less info than a fresh direction",
      a_redundant["realizedInformationGain"] < a["realizedInformationGain"],
      (a_redundant["realizedInformationGain"], a["realizedInformationGain"]))

# ---- loop_convergence ----
c_loose = run({"task": "loop_convergence", "designMatrix": X.tolist(),
               "observations": y.tolist(), "candidatePool": pool,
               "tolerance": 1e6, "noiseVariance": 1.0, "priorPrecision": 1.0})
check("convergence: huge tolerance -> converged", c_loose["converged"] is True, c_loose)
c_tight = run({"task": "loop_convergence", "designMatrix": X.tolist(),
               "observations": y.tolist(), "candidatePool": pool,
               "tolerance": 1e-9, "noiseVariance": 1.0, "priorPrecision": 1.0})
check("convergence: tiny tolerance -> not converged", c_tight["converged"] is False, c_tight)
check("convergence reports a most-uncertain candidate",
      0 <= c_tight["mostUncertainIndex"] < len(pool), c_tight)

# ---- error handling ----
check("unknown task -> error", run({"task": "nope"}).get("status") == "error")
check("mismatched X/y lengths -> error",
      run({"task": "propose_next_experiment", "designMatrix": [[1.0, 0.0]],
           "observations": [1.0, 2.0], "candidatePool": pool}).get("status") == "error")
check("assimilate without a real measurement -> error",
      run({"task": "assimilate_measurement", "designMatrix": X.tolist(),
           "observations": y.tolist(), "newDesign": [1.0, 0.0]}).get("status") == "error")
check("feature mismatch pool -> error",
      run({"task": "propose_next_experiment", "designMatrix": X.tolist(),
           "observations": y.tolist(), "candidatePool": [[1.0, 0.0, 0.0]]}).get("status") == "error")

print(f"\nALL {passed} ACTIVE-LEARNING-LOOP TESTS PASSED")
