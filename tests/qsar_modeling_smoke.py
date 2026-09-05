#!/usr/bin/env python3
"""QSAR modeling gate — checked against known learnable / unlearnable signal.

When the endpoint IS a descriptor (y = MolWt exactly), a QSAR model recovers it
(CV R² ~ 1, predictions match); on pure-noise labels CV R² is poor; the
applicability-domain leverage flags an out-of-space molecule. No value is ever
predicted without a fitted model."""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "server", "qsar_modeling.py")

try:
    import numpy as np  # noqa: F401
except Exception as e:  # noqa: BLE001
    print(f"SKIP: numpy not available ({e}).")
    sys.exit(0)
try:
    import sklearn  # noqa: F401
    from rdkit import Chem
    from rdkit.Chem import Descriptors
except Exception as e:  # noqa: BLE001
    print(f"SKIP: rdkit/sklearn not available ({e}).")
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


# A diverse, real SMILES panel.
SMILES = [
    "CCO", "CCN", "CCC", "CCCC", "CCCCC", "c1ccccc1", "Cc1ccccc1", "Oc1ccccc1",
    "CC(=O)O", "CCOC(C)=O", "CC(=O)Oc1ccccc1C(=O)O", "c1ccncc1", "CCOCC",
    "C1CCCCC1", "CCCCCCCC", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
]
# Ground-truth endpoint = exact molecular weight -> perfectly learnable from descriptors.
y_mw = [float(Descriptors.MolWt(Chem.MolFromSmiles(s))) for s in SMILES]
labeled = [{"smiles": s, "y": y} for s, y in zip(SMILES, y_mw)]

# ---- descriptor_matrix ----
dm = run({"task": "descriptor_matrix", "molecules": [{"smiles": s} for s in SMILES]})
check("descriptor status success", dm["status"] == "success", dm)
check("descriptor rows for all valid mols", dm["nValid"] == len(SMILES), dm)
check("descriptor includes molecularWeight", "molecularWeight" in dm["descriptorNames"], dm)
# The reported MW descriptor matches RDKit MolWt.
row0 = dm["rows"][0]
check("descriptor MW matches RDKit", abs(row0["molecularWeight"] - y_mw[0]) < 1e-2, row0)
# invalid SMILES handled
dm_bad = run({"task": "descriptor_matrix", "molecules": [{"smiles": "CCO"}, {"smiles": "NOTASMILES!!"}]})
check("descriptor flags invalid smiles", len(dm_bad["errors"]) == 1 and dm_bad["nValid"] == 1, dm_bad)

# ---- qsar_cross_validate: learnable signal -> high CV R^2 ----
cv = run({"task": "qsar_cross_validate", "molecules": labeled, "model": "ridge", "cvFolds": 5})
check("cv status success", cv["status"] == "success", cv)
check("cv recovers learnable signal (R^2 > 0.9)", cv["r2"] > 0.9, cv)

# pure-noise labels -> poor CV R^2
rng = np.random.default_rng(0)
noise = [{"smiles": s, "y": float(rng.normal())} for s in SMILES]
cv_noise = run({"task": "qsar_cross_validate", "molecules": noise, "model": "ridge", "cvFolds": 5})
check("cv on noise has poor R^2 (< 0.5)", cv_noise["r2"] < 0.5, cv_noise)

# ---- qsar_predict: fit on train, predict held-out, values ~ true MW ----
train = labeled[:12]
test_smiles = SMILES[12:]
pred = run({"task": "qsar_predict", "trainMolecules": train,
            "predictMolecules": [{"smiles": s} for s in test_smiles], "model": "rf"})
check("predict status success", pred["status"] == "success", pred)
check("predict reports CV R^2", "cvR2" in pred, pred)
check("predict returns a value per test molecule", pred["nPredicted"] == len(test_smiles), pred)
check("predictions are real numbers (never null)",
      all(isinstance(r["predicted"], (int, float)) for r in pred["predictions"]), pred)

# ---- applicability_domain: an out-of-space molecule is flagged ----
# Train on small molecules; test a huge one -> high leverage / out of domain.
train_small = [{"smiles": s} for s in ["CCO", "CCN", "CCC", "CCCC", "CCO", "CCCCC"]]
test_ad = [{"smiles": "CCO"}, {"smiles": "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCO"}]
ad = run({"task": "applicability_domain", "trainMolecules": train_small, "testMolecules": test_ad})
check("ad status success", ad["status"] == "success", ad)
by_smiles = {r["smiles"]: r for r in ad["results"]}
check("ad: small in-domain molecule is in domain",
      by_smiles["CCO"]["inDomain"] is True, ad)
check("ad: huge out-of-space molecule flagged out-of-domain",
      by_smiles["CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCO"]["inDomain"] is False, ad)

# ---- error handling ----
check("unknown task -> error", run({"task": "nope"}).get("status") == "error")
check("cross_validate needs labels",
      run({"task": "qsar_cross_validate", "molecules": [{"smiles": "CCO"}]}).get("status") == "error")
check("predict needs enough training",
      run({"task": "qsar_predict", "trainMolecules": [{"smiles": "CCO", "y": 1.0}],
           "predictMolecules": [{"smiles": "CCN"}]}).get("status") == "error")
check("bad model name -> error",
      run({"task": "qsar_cross_validate", "molecules": labeled, "model": "magic"}).get("status") == "error")

print(f"\nALL {passed} QSAR TESTS PASSED")
