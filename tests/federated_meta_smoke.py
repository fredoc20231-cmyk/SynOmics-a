#!/usr/bin/env python3
"""Federated meta-analysis gate — checked against exact references.

The federated t-test (per-site sufficient stats only) must EQUAL scipy's Welch
t-test on the concatenated raw data. Stouffer/Fisher combinations match closed
forms; DerSimonian–Laird gives τ²=0 / I²=0 for identical sites and a tighter CI
than any single site."""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "server", "federated_meta.py")

try:
    import numpy as np
    from scipy import stats
except Exception as e:  # noqa: BLE001
    print(f"SKIP: numpy/scipy not available ({e}).")
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


def suff(x):
    x = np.asarray(x, float)
    return len(x), float(np.mean(x)), float(np.var(x, ddof=1))


# --------------------------------------------------------------------------- #
# Task 1 — federated Welch t-test EQUALS the pooled-raw-data Welch t-test.
# --------------------------------------------------------------------------- #
rng = np.random.default_rng(0)
# Three sites, each with group A and B raw data (kept local; only stats shared).
sitesA = [rng.normal(10.0, 2.0, 20), rng.normal(10.0, 2.0, 25), rng.normal(10.0, 2.0, 18)]
sitesB = [rng.normal(12.0, 3.0, 22), rng.normal(12.0, 3.0, 19), rng.normal(12.0, 3.0, 24)]
sites = []
for a, b in zip(sitesA, sitesB):
    nA, mA, vA = suff(a)
    nB, mB, vB = suff(b)
    sites.append({"nA": nA, "meanA": mA, "varA": vA, "nB": nB, "meanB": mB, "varB": vB})

allA = np.concatenate(sitesA)
allB = np.concatenate(sitesB)
t_ref, p_ref = stats.ttest_ind(allA, allB, equal_var=False)

d = run({"task": "federated_ttest", "sites": sites})
check("ftest status success", d["status"] == "success", d)
check("federated t EXACTLY matches pooled-raw Welch t", abs(d["t"] - float(t_ref)) < 1e-9,
      (d["t"], float(t_ref)))
check("federated p EXACTLY matches pooled-raw Welch p", abs(d["pValue"] - float(p_ref)) < 1e-9,
      (d["pValue"], float(p_ref)))
check("federated pooled N(A) equals total raw N(A)", d["pooledA"]["n"] == len(allA), d["pooledA"])
check("federated pooled mean(A) matches raw mean(A)",
      abs(d["pooledA"]["mean"] - float(np.mean(allA))) < 1e-9, d["pooledA"])
check("federated pooled var(A) matches raw var(A)",
      abs(d["pooledA"]["variance"] - float(np.var(allA, ddof=1))) < 1e-9, d["pooledA"])

# --------------------------------------------------------------------------- #
# Task 2 — Stouffer combination.
# --------------------------------------------------------------------------- #
z = [2.0, 2.0, 2.0, 2.0]  # unweighted -> combined = sum/sqrt(k) = 8/2 = 4.0
s = run({"task": "stouffer_meta", "zScores": z})
check("stouffer status success", s["status"] == "success", s)
check("stouffer combined Z = 4.0 for four z=2", abs(s["combinedZ"] - 4.0) < 1e-9, s)
# single z -> unchanged
s1 = run({"task": "stouffer_meta", "zScores": [1.5]})
check("stouffer single z unchanged", abs(s1["combinedZ"] - 1.5) < 1e-9, s1)

# --------------------------------------------------------------------------- #
# Task 3 — Fisher combination matches scipy.combine_pvalues.
# --------------------------------------------------------------------------- #
pv = [0.01, 0.2, 0.05, 0.5]
f = run({"task": "fisher_meta", "pValues": pv})
chi_ref, p_ref_f = stats.combine_pvalues(pv, method="fisher")
check("fisher status success", f["status"] == "success", f)
check("fisher chi2 matches scipy", abs(f["chiSquare"] - float(chi_ref)) < 1e-9, (f["chiSquare"], float(chi_ref)))
check("fisher combined p matches scipy", abs(f["combinedPValue"] - float(p_ref_f)) < 1e-9,
      (f["combinedPValue"], float(p_ref_f)))
check("fisher df = 2k", f["degreesOfFreedom"] == 2 * len(pv), f)

# --------------------------------------------------------------------------- #
# Task 4 — random-effects: identical sites -> tau2=0, I2=0, pooled=effect.
# --------------------------------------------------------------------------- #
re_ident = run({"task": "random_effects_meta", "effects": [0.5, 0.5, 0.5],
                "standardErrors": [0.1, 0.1, 0.1]})
check("re status success", re_ident["status"] == "success", re_ident)
check("re identical sites -> tau2 ~ 0", abs(re_ident["tau2"]) < 1e-9, re_ident)
check("re identical sites -> I2 ~ 0", abs(re_ident["I2Percent"]) < 1e-6, re_ident)
check("re identical sites -> pooled = 0.5", abs(re_ident["pooledEffect"] - 0.5) < 1e-9, re_ident)
# pooled SE is tighter than any single site's SE (0.1)
check("re pooled SE < single-site SE", re_ident["pooledSE"] < 0.1, re_ident)
# heterogeneous sites -> positive tau2 / I2
re_het = run({"task": "random_effects_meta", "effects": [0.1, 0.9, 0.2, 1.2],
              "standardErrors": [0.1, 0.1, 0.1, 0.1]})
check("re heterogeneous -> I2 > 0", re_het["I2Percent"] > 0, re_het)
check("re heterogeneous -> tau2 > 0", re_het["tau2"] > 0, re_het)

# --------------------------------------------------------------------------- #
# Error handling.
# --------------------------------------------------------------------------- #
check("unknown task -> error", run({"task": "nope"}).get("status") == "error")
check("ftest missing sites -> error", run({"task": "federated_ttest"}).get("status") == "error")
check("fisher rejects p>1", run({"task": "fisher_meta", "pValues": [1.5]}).get("status") == "error")
check("re needs >=2 sites",
      run({"task": "random_effects_meta", "effects": [0.5], "standardErrors": [0.1]}).get("status") == "error")

print(f"\nALL {passed} FEDERATED-META TESTS PASSED")
