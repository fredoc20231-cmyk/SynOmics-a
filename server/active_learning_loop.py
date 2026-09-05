#!/usr/bin/env python3
"""Active-learning loop — the stateful "self-driving lab" brain (single dispatch).

Bridges the design layer (`experimental_design.py`) with real measurements: it
FITS a Bayesian linear model on the experiments run so far, proposes the next
experiment that maximally reduces model uncertainty, then HALTS for the real
measurement. It never invents an assay outcome — every observation `y` must be a
real measurement supplied by the caller. Reads JSON on stdin, prints JSON on
stdout.

ZERO HALLUCINATION: posterior means/variances and information gains are exact
linear algebra on the provided (X, y). The proposal step returns
`awaitingMeasurement` and NO fabricated response value; iteration only advances
when a real measurement is assimilated.

Tasks
-----
- propose_next_experiment : fit the posterior on (X, y), report current
  coefficient estimates + uncertainty, and select the next experiment from a
  candidate pool by expected information gain. Halts for a real measurement.
- assimilate_measurement  : append a newly MEASURED (x, y) to the data, refit,
  and report the realized information gain and posterior-variance reduction.
- loop_convergence        : given the data and a candidate pool, report whether
  the maximum predictive uncertainty over the pool has fallen below a tolerance
  (the honest stop criterion for the loop).

Bayesian linear regression (Gaussian prior w ~ N(0, τ⁻¹I), noise σ²):
  A = τI + XᵀX/σ²      (posterior precision)
  Σ = A⁻¹              (posterior covariance)
  μ = Σ Xᵀ y / σ²      (posterior mean of the coefficients)
  predictive variance at x:  xᵀΣx  (model term; +σ² for a new observation)
"""
import json
import sys


def _fail(msg, status="error"):
    print(json.dumps({"status": status, "error": msg}))
    sys.exit(0)


def _load2d(np, value, name):
    try:
        M = np.asarray(value, float)
    except Exception as e:
        _fail(f"`{name}` must be numeric: {e}")
    if M.ndim != 2:
        _fail(f"`{name}` must be 2D; got ndim={M.ndim}.")
    if not np.all(np.isfinite(M)):
        _fail(f"`{name}` must contain only finite numbers.")
    return M


def _load1d(np, value, name):
    try:
        v = np.asarray(value, float).ravel()
    except Exception as e:
        _fail(f"`{name}` must be numeric: {e}")
    if not np.all(np.isfinite(v)):
        _fail(f"`{name}` must contain only finite numbers.")
    return v


def _fit_posterior(np, X, y, noise_var, prior_precision):
    """Return (mu, Sigma, A) for Bayesian linear regression on (X, y)."""
    d = X.shape[1]
    A = prior_precision * np.eye(d)
    if X.shape[0] > 0:
        A = A + (X.T @ X) / noise_var
    Sigma = np.linalg.inv(A)
    if X.shape[0] > 0:
        mu = Sigma @ (X.T @ y) / noise_var
    else:
        mu = np.zeros(d)
    return mu, Sigma, A


def _params(np, p):
    try:
        noise_var = float(p.get("noiseVariance", 1.0))
        prior_precision = float(p.get("priorPrecision", 1.0))
    except Exception:
        _fail("`noiseVariance` and `priorPrecision` must be numbers.")
    if noise_var <= 0 or prior_precision <= 0:
        _fail("`noiseVariance` and `priorPrecision` must be > 0.")
    return noise_var, prior_precision


def _get_Xy(np, p):
    X = _load2d(np, p.get("designMatrix"), "designMatrix")
    y = _load1d(np, p.get("observations"), "observations")
    if X.shape[0] != y.shape[0]:
        _fail(
            f"`designMatrix` has {X.shape[0]} rows but `observations` has "
            f"{y.shape[0]} — they must correspond one-to-one (real measurements)."
        )
    return X, y


# --------------------------------------------------------------------------- #
# Task 1 — propose the next experiment (halts for a real measurement)
# --------------------------------------------------------------------------- #
def task_propose_next_experiment(p):
    try:
        import numpy as np
    except Exception as e:  # pragma: no cover - environment dependent
        _fail(f"propose_next_experiment requires numpy: {e}", status="unavailable")

    if p.get("designMatrix") is None or p.get("observations") is None:
        _fail("Provide `designMatrix` (experiments×features) and `observations` (measured y per experiment).")
    X, y = _get_Xy(np, p)
    d = X.shape[1]

    pool = p.get("candidatePool")
    if not isinstance(pool, list) or not pool:
        _fail("Provide `candidatePool` (2D array: candidates × features).")
    Xc = _load2d(np, pool, "candidatePool")
    if Xc.shape[1] != d:
        _fail(f"`candidatePool` has {Xc.shape[1]} features but the model has {d}.")

    noise_var, prior_precision = _params(np, p)
    mu, Sigma, _A = _fit_posterior(np, X, y, noise_var, prior_precision)

    # Rank candidates by expected information gain.
    ranked = []
    for i in range(Xc.shape[0]):
        x = Xc[i]
        pv = float(x @ Sigma @ x)
        eig = 0.5 * float(np.log1p(pv / noise_var))
        ranked.append({
            "index": i,
            "expectedInformationGain": round(eig, 10),
            "predictedMean": round(float(x @ mu), 8),
            "predictiveStd": round(float(np.sqrt(pv + noise_var)), 8),
        })
    ranked.sort(key=lambda r: -r["expectedInformationGain"])
    best = ranked[0]

    coef = [
        {"index": j, "estimate": round(float(mu[j]), 8),
         "std": round(float(np.sqrt(Sigma[j, j])), 8)}
        for j in range(d)
    ]

    analysis = (
        f"Fitted a Bayesian linear model on {X.shape[0]} measured experiment(s) "
        f"({d} coefficient(s)); selected candidate #{best['index']} as the next "
        f"experiment (EIG={best['expectedInformationGain']:.4g} nats). AWAITING a "
        f"real measurement — no response value is fabricated."
    )
    research_log = (
        "# Active-learning loop — next experiment proposal\n\n"
        f"The posterior over the {d} model coefficient(s) was fit on the "
        f"{X.shape[0]} experiment(s) run so far. The candidate with the largest "
        "expected information gain is proposed as the next experiment. The loop "
        "then **halts**: the caller must run that experiment and supply the real "
        "measured response via `assimilate_measurement` — the loop never invents "
        "an outcome.\n\n"
        f"| Field | Value |\n| --- | --- |\n"
        f"| Experiments run | {X.shape[0]} |\n"
        f"| Coefficients | {d} |\n"
        f"| Proposed candidate | #{best['index']} |\n"
        f"| Proposed EIG (nats) | {best['expectedInformationGain']:.4g} |\n"
        f"| Predicted response (prior to measuring) | {best['predictedMean']:.4g} "
        f"± {best['predictiveStd']:.4g} |\n"
    )
    return {
        "status": "success",
        "awaitingMeasurement": True,
        "analysis": analysis,
        "proposedIndex": best["index"],
        "proposedDesign": [round(float(v), 8) for v in Xc[best["index"]]],
        "ranked": ranked,
        "coefficients": coef,
        "nObserved": int(X.shape[0]),
        "nFeatures": d,
        "researchLog": research_log,
    }


# --------------------------------------------------------------------------- #
# Task 2 — assimilate a real measurement and refit
# --------------------------------------------------------------------------- #
def task_assimilate_measurement(p):
    try:
        import numpy as np
    except Exception as e:  # pragma: no cover - environment dependent
        _fail(f"assimilate_measurement requires numpy: {e}", status="unavailable")

    if p.get("designMatrix") is None or p.get("observations") is None:
        _fail("Provide the prior `designMatrix` and `observations`.")
    X, y = _get_Xy(np, p)
    d = X.shape[1]

    new_x = p.get("newDesign")
    new_y = p.get("newObservation")
    if new_x is None or new_y is None:
        _fail("Provide `newDesign` (the run experiment's design row) and `newObservation` (its REAL measured response).")
    xn = _load1d(np, new_x, "newDesign")
    if xn.shape[0] != d:
        _fail(f"`newDesign` has {xn.shape[0]} features but the model has {d}.")
    try:
        yn = float(new_y)
    except Exception:
        _fail("`newObservation` must be a single real measured number.")
    if not np.isfinite(yn):
        _fail("`newObservation` must be finite.")

    noise_var, prior_precision = _params(np, p)
    mu_old, Sigma_old, A_old = _fit_posterior(np, X, y, noise_var, prior_precision)

    X_new = np.vstack([X, xn.reshape(1, -1)])
    y_new = np.concatenate([y, [yn]])
    mu_new, Sigma_new, A_new = _fit_posterior(np, X_new, y_new, noise_var, prior_precision)

    trace_old = float(np.trace(Sigma_old))
    trace_new = float(np.trace(Sigma_new))
    trace_reduction_pct = (
        100.0 * (trace_old - trace_new) / trace_old if trace_old > 0 else 0.0
    )
    # Realized information gain (nats) = 0.5 * (logdet A_new - logdet A_old).
    info_gain = 0.5 * float(np.linalg.slogdet(A_new)[1] - np.linalg.slogdet(A_old)[1])

    coef = [
        {"index": j,
         "estimateBefore": round(float(mu_old[j]), 8),
         "estimateAfter": round(float(mu_new[j]), 8),
         "stdBefore": round(float(np.sqrt(Sigma_old[j, j])), 8),
         "stdAfter": round(float(np.sqrt(Sigma_new[j, j])), 8)}
        for j in range(d)
    ]
    analysis = (
        f"Assimilated 1 real measurement (y={yn:g}); model refit on "
        f"{X_new.shape[0]} experiment(s). Realized information gain "
        f"{info_gain:.4g} nats; posterior variance (trace) reduced by "
        f"{trace_reduction_pct:.3g}%."
    )
    research_log = (
        "# Active-learning loop — measurement assimilation\n\n"
        "The newly measured experiment was appended to the data and the Bayesian "
        "posterior refit. The realized information gain is "
        "½·(log det A_new − log det A_old); the trace of the posterior covariance "
        "quantifies total remaining parameter uncertainty.\n\n"
        f"| Metric | Value |\n| --- | --- |\n"
        f"| Experiments (after) | {X_new.shape[0]} |\n"
        f"| Realized info gain (nats) | {info_gain:.4g} |\n"
        f"| Posterior trace before | {trace_old:.6g} |\n"
        f"| Posterior trace after | {trace_new:.6g} |\n"
        f"| Variance reduction | {trace_reduction_pct:.3g}% |\n"
    )
    return {
        "status": "success",
        "analysis": analysis,
        "realizedInformationGain": round(info_gain, 10),
        "posteriorTraceBefore": round(trace_old, 10),
        "posteriorTraceAfter": round(trace_new, 10),
        "traceReductionPercent": round(trace_reduction_pct, 6),
        "coefficients": coef,
        "nObserved": int(X_new.shape[0]),
        "nFeatures": d,
        "researchLog": research_log,
    }


# --------------------------------------------------------------------------- #
# Task 3 — loop convergence / stop criterion
# --------------------------------------------------------------------------- #
def task_loop_convergence(p):
    try:
        import numpy as np
    except Exception as e:  # pragma: no cover - environment dependent
        _fail(f"loop_convergence requires numpy: {e}", status="unavailable")

    if p.get("designMatrix") is None or p.get("observations") is None:
        _fail("Provide `designMatrix` and `observations`.")
    X, y = _get_Xy(np, p)
    d = X.shape[1]
    pool = p.get("candidatePool")
    if not isinstance(pool, list) or not pool:
        _fail("Provide `candidatePool` (2D array: candidates × features).")
    Xc = _load2d(np, pool, "candidatePool")
    if Xc.shape[1] != d:
        _fail(f"`candidatePool` has {Xc.shape[1]} features but the model has {d}.")
    try:
        tol = float(p.get("tolerance"))
    except Exception:
        _fail("Provide a numeric `tolerance` (max acceptable predictive std).")
    if tol <= 0:
        _fail("`tolerance` must be > 0.")

    noise_var, prior_precision = _params(np, p)
    _mu, Sigma, _A = _fit_posterior(np, X, y, noise_var, prior_precision)

    stds = []
    for i in range(Xc.shape[0]):
        x = Xc[i]
        pv = float(x @ Sigma @ x)
        stds.append(float(np.sqrt(pv + noise_var)))
    stds = np.asarray(stds)
    max_i = int(np.argmax(stds))
    max_std = float(stds[max_i])
    converged = bool(max_std <= tol)

    analysis = (
        f"Loop {'CONVERGED' if converged else 'NOT converged'}: maximum predictive "
        f"std over {Xc.shape[0]} candidate(s) is {max_std:.4g} "
        f"({'≤' if converged else '>'} tolerance {tol:g}). "
        + ("No candidate exceeds the uncertainty tolerance — stop."
           if converged else
           f"Candidate #{max_i} remains the most uncertain — continue.")
    )
    research_log = (
        "# Active-learning loop — convergence check\n\n"
        "The loop stops when no remaining candidate has predictive uncertainty "
        "above the tolerance (further experiments would add little information).\n\n"
        f"| Metric | Value |\n| --- | --- |\n"
        f"| Experiments run | {X.shape[0]} |\n"
        f"| Max predictive std | {max_std:.4g} |\n"
        f"| Tolerance | {tol:g} |\n"
        f"| Converged | {converged} |\n"
        f"| Most-uncertain candidate | #{max_i} |\n"
    )
    return {
        "status": "success",
        "analysis": analysis,
        "converged": converged,
        "maxPredictiveStd": round(max_std, 10),
        "mostUncertainIndex": max_i,
        "tolerance": tol,
        "nObserved": int(X.shape[0]),
        "researchLog": research_log,
    }


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
TASKS = {
    "propose_next_experiment": task_propose_next_experiment,
    "assimilate_measurement": task_assimilate_measurement,
    "loop_convergence": task_loop_convergence,
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
