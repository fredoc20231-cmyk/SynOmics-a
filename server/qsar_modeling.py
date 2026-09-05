#!/usr/bin/env python3
"""QSAR / QSPR modeling — real cross-validated property prediction (dispatch).

Trains a structure-property regression on the caller's REAL measured data
(SMILES + a measured endpoint) using RDKit descriptors + scikit-learn, and
predicts new molecules ONLY from that fitted model. Reads JSON on stdin, prints
JSON on stdout.

ZERO HALLUCINATION: this is the exact opposite of a magic-number predictor. No
activity/property value is ever invented — every prediction comes from a model
fit on the user's real training data, and the model's honesty is reported as
k-fold cross-validated performance (R², RMSE, MAE). If no training data with real
labels is supplied, the tools error rather than guess. An applicability-domain
check flags predictions the model should not be trusted for.

Tasks
-----
- descriptor_matrix   : compute the RDKit descriptor table for a set of SMILES.
- qsar_cross_validate : k-fold CV performance (R²/RMSE/MAE) of a model on labeled
  training molecules — the honest reliability estimate.
- qsar_predict        : fit on labeled training molecules, predict unlabeled
  molecules, and report the model's CV performance alongside the predictions.
- applicability_domain: leverage-based flag for test molecules outside the
  training descriptor space (predictions there are extrapolations).
"""
import json
import sys

# The real RDKit descriptor panel used as QSAR features (all deterministic).
_DESCRIPTORS = [
    "molecularWeight", "logP", "tpsa", "hBondDonors", "hBondAcceptors",
    "rotatableBonds", "aromaticRings", "fractionCsp3", "molarRefractivity",
    "heavyAtoms", "numRings",
]


def _fail(msg, status="error"):
    print(json.dumps({"status": status, "error": msg}))
    sys.exit(0)


def _descriptors_for(mol):
    """Return the real RDKit descriptor vector (order = _DESCRIPTORS)."""
    from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

    return [
        float(Descriptors.MolWt(mol)),
        float(Crippen.MolLogP(mol)),
        float(rdMolDescriptors.CalcTPSA(mol)),
        float(Lipinski.NumHDonors(mol)),
        float(Lipinski.NumHAcceptors(mol)),
        float(Lipinski.NumRotatableBonds(mol)),
        float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        float(rdMolDescriptors.CalcFractionCSP3(mol)),
        float(Crippen.MolMR(mol)),
        float(mol.GetNumHeavyAtoms()),
        float(rdMolDescriptors.CalcNumRings(mol)),
    ]


def _featurize(smiles_list):
    """Return (X list, valid indices, errors) — invalid SMILES are skipped honestly."""
    from rdkit import Chem

    X, valid, errors = [], [], []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            errors.append({"index": i, "smiles": smi, "error": "unparsable SMILES"})
            continue
        X.append(_descriptors_for(mol))
        valid.append(i)
    return X, valid, errors


def _parse_labeled(np, mols, need_y=True):
    """Parse [{smiles, y}] -> (smiles, y ndarray or None, errors)."""
    if not isinstance(mols, list) or not mols:
        _fail("Provide a non-empty list of molecules.")
    smiles, ys = [], []
    for k, m in enumerate(mols):
        if not isinstance(m, dict) or "smiles" not in m:
            _fail(f"Molecule #{k} must be an object with a `smiles` field.")
        smiles.append(m["smiles"])
        if need_y:
            if "y" not in m:
                _fail(f"Molecule #{k} must have a real measured `y` (endpoint value).")
            try:
                ys.append(float(m["y"]))
            except Exception:
                _fail(f"Molecule #{k} `y` must be numeric.")
    y = np.asarray(ys, float) if need_y else None
    if need_y and not np.all(np.isfinite(y)):
        _fail("All `y` values must be finite real measurements.")
    return smiles, y


def _make_model(np, name):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    name = str(name).lower()
    if name == "rf":
        return RandomForestRegressor(n_estimators=200, random_state=0), False
    if name == "ridge":
        return Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=1.0))]), True
    _fail("`model` must be 'ridge' or 'rf'.")


# --------------------------------------------------------------------------- #
# Task 1 — descriptor matrix
# --------------------------------------------------------------------------- #
def task_descriptor_matrix(p):
    try:
        from rdkit import Chem  # noqa: F401
    except Exception as e:  # pragma: no cover - environment dependent
        _fail(f"descriptor_matrix requires rdkit: {e}", status="unavailable")

    mols = p.get("molecules")
    if not isinstance(mols, list) or not mols:
        _fail("Provide `molecules` as [{smiles}, ...].")
    smiles = [m.get("smiles") if isinstance(m, dict) else m for m in mols]
    X, valid, errors = _featurize(smiles)
    rows = [
        {"index": valid[j], "smiles": smiles[valid[j]],
         **{name: round(X[j][c], 6) for c, name in enumerate(_DESCRIPTORS)}}
        for j in range(len(valid))
    ]
    analysis = (
        f"Computed {len(_DESCRIPTORS)} RDKit descriptor(s) for {len(valid)} of "
        f"{len(smiles)} molecule(s)"
        + (f"; {len(errors)} unparsable." if errors else ".")
    )
    return {
        "status": "success",
        "analysis": analysis,
        "descriptorNames": _DESCRIPTORS,
        "rows": rows,
        "errors": errors,
        "nValid": len(valid),
        "researchLog": (
            "# QSAR descriptor matrix\n\n"
            f"Each molecule was parsed with RDKit and described by "
            f"{len(_DESCRIPTORS)} deterministic physicochemical descriptors "
            f"({', '.join(_DESCRIPTORS)}). Unparsable SMILES are skipped, not "
            "fabricated.\n"
        ),
    }


# --------------------------------------------------------------------------- #
# Task 2 — cross-validated performance
# --------------------------------------------------------------------------- #
def task_qsar_cross_validate(p):
    try:
        import numpy as np
    except Exception as e:  # pragma: no cover - environment dependent
        _fail(f"qsar_cross_validate requires numpy: {e}", status="unavailable")
    try:
        from rdkit import Chem  # noqa: F401
        from sklearn.model_selection import KFold, cross_val_predict
    except Exception as e:  # pragma: no cover - environment dependent
        _fail(f"qsar_cross_validate requires rdkit+scikit-learn: {e}", status="unavailable")

    smiles, y_all = _parse_labeled(np, p.get("molecules"))
    X_list, valid, errors = _featurize(smiles)
    if len(valid) < 4:
        _fail("Need at least 4 molecules with valid SMILES + labels for cross-validation.")
    X = np.asarray(X_list, float)
    y = y_all[valid]
    model, _needs_scale = _make_model(np, p.get("model", "ridge"))
    try:
        folds = int(p.get("cvFolds", 5))
    except Exception:
        _fail("`cvFolds` must be an integer.")
    folds = max(2, min(folds, len(valid)))

    kf = KFold(n_splits=folds, shuffle=True, random_state=0)
    y_pred = cross_val_predict(model, X, y, cv=kf)
    resid = y - y_pred
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(resid**2)))
    mae = float(np.mean(np.abs(resid)))

    analysis = (
        f"{folds}-fold cross-validation of a {p.get('model', 'ridge')} QSAR model "
        f"on {len(valid)} molecule(s): CV R²={r2:.4g}, RMSE={rmse:.4g}, "
        f"MAE={mae:.4g}. This is the honest out-of-fold reliability — no value is "
        f"predicted outside a fitted model."
    )
    return {
        "status": "success",
        "analysis": analysis,
        "model": str(p.get("model", "ridge")),
        "cvFolds": folds,
        "r2": round(float(r2), 8),
        "rmse": round(rmse, 8),
        "mae": round(mae, 8),
        "nTrain": len(valid),
        "errors": errors,
        "researchLog": (
            "# QSAR cross-validation\n\n"
            f"A {p.get('model', 'ridge')} regressor on {len(_DESCRIPTORS)} RDKit "
            f"descriptors was evaluated by {folds}-fold cross-validation "
            "(shuffle, seed 0). Out-of-fold predictions give an unbiased estimate "
            "of predictive performance.\n\n"
            f"| Metric | Value |\n| --- | --- |\n| CV R² | {r2:.4g} |\n"
            f"| RMSE | {rmse:.4g} |\n| MAE | {mae:.4g} |\n| Molecules | {len(valid)} |\n"
        ),
    }


# --------------------------------------------------------------------------- #
# Task 3 — fit + predict
# --------------------------------------------------------------------------- #
def task_qsar_predict(p):
    try:
        import numpy as np
    except Exception as e:  # pragma: no cover - environment dependent
        _fail(f"qsar_predict requires numpy: {e}", status="unavailable")
    try:
        from rdkit import Chem  # noqa: F401
        from sklearn.model_selection import KFold, cross_val_predict
    except Exception as e:  # pragma: no cover - environment dependent
        _fail(f"qsar_predict requires rdkit+scikit-learn: {e}", status="unavailable")

    train = p.get("trainMolecules")
    test = p.get("predictMolecules")
    smiles_tr, y_tr_all = _parse_labeled(np, train)
    if not isinstance(test, list) or not test:
        _fail("Provide `predictMolecules` as [{smiles}, ...] to predict.")
    smiles_te = [m.get("smiles") if isinstance(m, dict) else m for m in test]

    Xtr_list, valid_tr, err_tr = _featurize(smiles_tr)
    if len(valid_tr) < 4:
        _fail("Need at least 4 valid labeled training molecules.")
    Xte_list, valid_te, err_te = _featurize(smiles_te)
    if not valid_te:
        _fail("No valid molecules to predict.")
    Xtr = np.asarray(Xtr_list, float)
    ytr = y_tr_all[valid_tr]
    Xte = np.asarray(Xte_list, float)

    model, _ns = _make_model(np, p.get("model", "ridge"))
    # Honest reliability: cross-validated R² on the training set.
    folds = max(2, min(5, len(valid_tr)))
    y_cv = cross_val_predict(model, Xtr, ytr, cv=KFold(n_splits=folds, shuffle=True, random_state=0))
    ss_res = float(np.sum((ytr - y_cv) ** 2))
    ss_tot = float(np.sum((ytr - np.mean(ytr)) ** 2))
    cv_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    model.fit(Xtr, ytr)
    preds = model.predict(Xte)

    predictions = [
        {"index": valid_te[j], "smiles": smiles_te[valid_te[j]],
         "predicted": round(float(preds[j]), 8)}
        for j in range(len(valid_te))
    ]
    analysis = (
        f"Trained a {p.get('model', 'ridge')} QSAR model on {len(valid_tr)} "
        f"labeled molecule(s) (CV R²={cv_r2:.4g}) and predicted {len(valid_te)} "
        f"molecule(s). Predictions come only from the fitted model on real data."
    )
    return {
        "status": "success",
        "analysis": analysis,
        "model": str(p.get("model", "ridge")),
        "cvR2": round(float(cv_r2), 8),
        "predictions": predictions,
        "nTrain": len(valid_tr),
        "nPredicted": len(valid_te),
        "trainErrors": err_tr,
        "predictErrors": err_te,
        "researchLog": (
            "# QSAR fit + prediction\n\n"
            f"A {p.get('model', 'ridge')} model was fit on {len(valid_tr)} real "
            f"labeled molecules and used to predict {len(valid_te)} molecule(s). "
            f"The training set's cross-validated R² ({cv_r2:.4g}) is reported so "
            "the predictions carry an honest reliability estimate. Nothing is "
            "predicted without a fitted model.\n"
        ),
    }


# --------------------------------------------------------------------------- #
# Task 4 — applicability domain (leverage)
# --------------------------------------------------------------------------- #
def task_applicability_domain(p):
    try:
        import numpy as np
    except Exception as e:  # pragma: no cover - environment dependent
        _fail(f"applicability_domain requires numpy: {e}", status="unavailable")
    try:
        from rdkit import Chem  # noqa: F401
    except Exception as e:  # pragma: no cover - environment dependent
        _fail(f"applicability_domain requires rdkit: {e}", status="unavailable")

    train = p.get("trainMolecules")
    test = p.get("testMolecules")
    if not isinstance(train, list) or not train:
        _fail("Provide `trainMolecules` as [{smiles}, ...].")
    if not isinstance(test, list) or not test:
        _fail("Provide `testMolecules` as [{smiles}, ...].")
    smiles_tr = [m.get("smiles") if isinstance(m, dict) else m for m in train]
    smiles_te = [m.get("smiles") if isinstance(m, dict) else m for m in test]

    Xtr_list, valid_tr, _e1 = _featurize(smiles_tr)
    Xte_list, valid_te, _e2 = _featurize(smiles_te)
    if len(valid_tr) < 3:
        _fail("Need at least 3 valid training molecules for a leverage domain.")
    if not valid_te:
        _fail("No valid test molecules.")
    Xtr = np.asarray(Xtr_list, float)
    Xte = np.asarray(Xte_list, float)

    # Standardize by training statistics; leverage h = x (XᵀX)⁻¹ xᵀ (hat diagonal).
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd == 0] = 1.0
    Ztr = (Xtr - mu) / sd
    Zte = (Xte - mu) / sd
    XtX_inv = np.linalg.pinv(Ztr.T @ Ztr)
    n, k = Ztr.shape
    h_star = 3.0 * (k + 1) / n  # standard QSAR leverage warning threshold

    results = []
    for j in range(len(valid_te)):
        x = Zte[j]
        h = float(x @ XtX_inv @ x)
        results.append({
            "index": valid_te[j],
            "smiles": smiles_te[valid_te[j]],
            "leverage": round(h, 8),
            "inDomain": bool(h <= h_star),
        })
    n_out = sum(1 for r in results if not r["inDomain"])
    analysis = (
        f"Applicability domain (leverage) over {len(valid_te)} test molecule(s) "
        f"against {len(valid_tr)} training molecule(s): warning threshold "
        f"h*={h_star:.4g}; {n_out} molecule(s) flagged as extrapolation "
        f"(out-of-domain — predictions there are unreliable)."
    )
    return {
        "status": "success",
        "analysis": analysis,
        "leverageThreshold": round(float(h_star), 8),
        "results": results,
        "nOutOfDomain": n_out,
        "nTrain": len(valid_tr),
        "researchLog": (
            "# QSAR applicability domain (leverage)\n\n"
            "Test molecules were standardized by the training descriptor "
            "statistics and their leverage h = xᵀ(ZᵀZ)⁻¹x computed. Molecules with "
            f"h above the standard threshold h* = 3(k+1)/n = {h_star:.4g} lie "
            "outside the training descriptor space; a QSAR prediction there is an "
            "extrapolation and should be treated as unreliable rather than "
            "trusted.\n"
        ),
    }


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
TASKS = {
    "descriptor_matrix": task_descriptor_matrix,
    "qsar_cross_validate": task_qsar_cross_validate,
    "qsar_predict": task_qsar_predict,
    "applicability_domain": task_applicability_domain,
}


def main():
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
    except Exception as e:
        _fail(f"Invalid JSON payload: {e}")
    task = payload.get("task")
    if task not in TASKS:
        _fail(f"Unknown task {task!r}. Available: {', '.join(TASKS)}.")
    print(json.dumps(TASKS[task](payload)))


if __name__ == "__main__":
    main()
