#!/usr/bin/env python3
"""Bayesian optimal experimental design / active learning — single dispatch.

The decision layer of a closed-loop ("self-driving lab") workflow: given the
experiments already run and a pool of candidate next experiments, pick the
experiment that maximally reduces model uncertainty. Reads a JSON payload on
stdin, prints a JSON result on stdout.

ZERO HALLUCINATION: every number is computed by real linear algebra on the
provided design matrices under a Bayesian linear model. Nothing about assay
outcomes is invented — this selects WHICH experiment to run, it does not predict
its result.

Tasks
-----
- bayesian_optimal_design   : rank a candidate pool by expected information gain
  (EIG) / D-optimality under a Bayesian linear model with a Gaussian prior; the
  best candidate is the most informative next experiment.
- sequential_active_learning: greedily choose a batch of experiments, updating
  the posterior after each pick (Sherman–Morrison), maximizing cumulative EIG.
- d_optimal_selection       : greedy D-optimal subset — choose k rows from a
  candidate pool maximizing log det(prior + X'X/σ²) (the classic DoE criterion).
- space_filling_design      : maximin Latin-Hypercube design over given bounds
  (scipy.stats.qmc) for initial screening; seeded and reproducible.

Math (Bayesian linear regression):
  posterior precision  A = τ·I + (1/σ²)·Xᵀ X       (τ = prior precision)
  posterior covariance Σ = A⁻¹
  for a candidate row x, the rank-1 determinant lemma gives
     D-gain(x)  = log(1 + xᵀΣx / σ²)
     EIG(x)     = ½·log(1 + xᵀΣx / σ²)   (nats; Gaussian information gain)
  after selecting x, Σ updates by Sherman–Morrison:
     Σ' = Σ − (Σ x xᵀ Σ) / (σ² + xᵀΣx)
"""
import json
import sys


def _fail(msg, status="error"):
    print(json.dumps({"status": status, "error": msg}))
    sys.exit(0)


def _posterior_precision(np, X, noise_var, prior_precision):
    """A = τI + XᵀX/σ² for the already-observed design matrix X (may be empty)."""
    d = X.shape[1]
    A = prior_precision * np.eye(d)
    if X.shape[0] > 0:
        A = A + (X.T @ X) / noise_var
    return A


def _load_matrix(np, value, name):
    try:
        M = np.asarray(value, float)
    except Exception as e:
        _fail(f"`{name}` must be numeric: {e}")
    if not np.all(np.isfinite(M)):
        _fail(f"`{name}` must contain only finite numbers.")
    return M


# --------------------------------------------------------------------------- #
# Task 1 — rank a candidate pool by expected information gain
# --------------------------------------------------------------------------- #
def task_bayesian_optimal_design(p):
    try:
        import numpy as np
    except Exception as e:  # pragma: no cover - environment dependent
        _fail(f"bayesian_optimal_design requires numpy: {e}", status="unavailable")

    pool = p.get("candidatePool")
    if not isinstance(pool, list) or not pool:
        _fail("Provide `candidatePool` (2D array: candidates x features).")
    Xc = _load_matrix(np, pool, "candidatePool")
    if Xc.ndim != 2:
        _fail(f"`candidatePool` must be 2D (candidates x features); got ndim={Xc.ndim}.")
    n_cand, d = Xc.shape

    design = p.get("designMatrix", [])
    if design:
        Xt = _load_matrix(np, design, "designMatrix")
        if Xt.ndim != 2:
            _fail(f"`designMatrix` must be 2D; got ndim={Xt.ndim}.")
        if Xt.shape[1] != d:
            _fail(
                f"`designMatrix` has {Xt.shape[1]} features but `candidatePool` has "
                f"{d}; feature dimensions must match."
            )
    else:
        Xt = np.empty((0, d))

    try:
        noise_var = float(p.get("noiseVariance", 1.0))
        prior_precision = float(p.get("priorPrecision", 1.0))
    except Exception:
        _fail("`noiseVariance` and `priorPrecision` must be numbers.")
    if noise_var <= 0 or prior_precision <= 0:
        _fail("`noiseVariance` and `priorPrecision` must be > 0.")

    A = _posterior_precision(np, Xt, noise_var, prior_precision)
    try:
        Sigma = np.linalg.inv(A)
    except np.linalg.LinAlgError as e:
        _fail(f"Posterior precision is singular: {e}")

    results = []
    for i in range(n_cand):
        x = Xc[i]
        pv = float(x @ Sigma @ x)  # predictive variance (model part)
        gain = float(np.log1p(pv / noise_var))  # D-optimality gain (rank-1 lemma)
        eig = 0.5 * gain  # Gaussian expected information gain (nats)
        results.append({
            "index": i,
            "expectedInformationGain": round(eig, 10),
            "dOptimalityGain": round(gain, 10),
            "predictiveVariance": round(pv, 10),
        })
    ranked = sorted(results, key=lambda r: -r["expectedInformationGain"])
    best = ranked[0]

    analysis = (
        f"Ranked {n_cand} candidate experiment(s) over {d} design feature(s) by "
        f"expected information gain under a Bayesian linear model "
        f"(σ²={noise_var:g}, prior precision τ={prior_precision:g}, "
        f"{Xt.shape[0]} experiment(s) already run). Best next experiment: "
        f"candidate #{best['index']} (EIG={best['expectedInformationGain']:.4g} nats)."
    )
    research_log = (
        "# Bayesian optimal experimental design\n\n"
        "Under a Bayesian linear model with Gaussian prior, the posterior "
        "covariance after the already-run experiments is Σ = (τI + XᵀX/σ²)⁻¹. For "
        "each candidate design row x the expected information gain is "
        "EIG(x) = ½·log(1 + xᵀΣx/σ²) (nats) — the reduction in posterior entropy "
        "from running that experiment. The most informative experiment is the one "
        "with the largest EIG.\n\n"
        f"| Metric | Value |\n| --- | --- |\n"
        f"| Candidates | {n_cand} |\n| Design features | {d} |\n"
        f"| Experiments already run | {Xt.shape[0]} |\n"
        f"| Best candidate | #{best['index']} |\n"
        f"| Best EIG (nats) | {best['expectedInformationGain']:.4g} |\n"
    )
    return {
        "status": "success",
        "analysis": analysis,
        "results": ranked,
        "bestIndex": best["index"],
        "nCandidates": n_cand,
        "nFeatures": d,
        "nObserved": int(Xt.shape[0]),
        "researchLog": research_log,
    }


# --------------------------------------------------------------------------- #
# Task 2 — greedy sequential active learning (batch selection)
# --------------------------------------------------------------------------- #
def task_sequential_active_learning(p):
    try:
        import numpy as np
    except Exception as e:  # pragma: no cover - environment dependent
        _fail(f"sequential_active_learning requires numpy: {e}", status="unavailable")

    pool = p.get("candidatePool")
    if not isinstance(pool, list) or not pool:
        _fail("Provide `candidatePool` (2D array: candidates x features).")
    Xc = _load_matrix(np, pool, "candidatePool")
    if Xc.ndim != 2:
        _fail(f"`candidatePool` must be 2D; got ndim={Xc.ndim}.")
    n_cand, d = Xc.shape

    try:
        n_batch = int(p.get("nBatch", min(3, n_cand)))
    except Exception:
        _fail("`nBatch` must be an integer.")
    if n_batch < 1:
        _fail("`nBatch` must be >= 1.")
    if n_batch > n_cand:
        _fail(f"`nBatch` ({n_batch}) cannot exceed the candidate pool size ({n_cand}).")

    design = p.get("designMatrix", [])
    if design:
        Xt = _load_matrix(np, design, "designMatrix")
        if Xt.ndim != 2 or Xt.shape[1] != d:
            _fail("`designMatrix` must be 2D with the same feature count as the pool.")
    else:
        Xt = np.empty((0, d))

    try:
        noise_var = float(p.get("noiseVariance", 1.0))
        prior_precision = float(p.get("priorPrecision", 1.0))
    except Exception:
        _fail("`noiseVariance` and `priorPrecision` must be numbers.")
    if noise_var <= 0 or prior_precision <= 0:
        _fail("`noiseVariance` and `priorPrecision` must be > 0.")

    A = _posterior_precision(np, Xt, noise_var, prior_precision)
    try:
        Sigma = np.linalg.inv(A)
    except np.linalg.LinAlgError as e:
        _fail(f"Posterior precision is singular: {e}")

    remaining = list(range(n_cand))
    picks = []
    cumulative_eig = 0.0
    for _step in range(n_batch):
        best_i = None
        best_eig = -np.inf
        best_pv = None
        for i in remaining:
            x = Xc[i]
            pv = float(x @ Sigma @ x)
            eig = 0.5 * float(np.log1p(pv / noise_var))
            if eig > best_eig:
                best_eig, best_i, best_pv = eig, i, pv
        # Commit the best pick and update the posterior (Sherman–Morrison).
        x = Xc[best_i]
        Sx = Sigma @ x
        Sigma = Sigma - np.outer(Sx, Sx) / (noise_var + best_pv)
        cumulative_eig += best_eig
        picks.append({
            "index": best_i,
            "step": len(picks) + 1,
            "expectedInformationGain": round(best_eig, 10),
            "predictiveVariance": round(best_pv, 10),
            "cumulativeEIG": round(cumulative_eig, 10),
        })
        remaining.remove(best_i)

    analysis = (
        f"Greedily selected {n_batch} experiment(s) from {n_cand} candidate(s) by "
        f"iterative maximum expected information gain (posterior updated after each "
        f"pick). Cumulative EIG = {cumulative_eig:.4g} nats. Selected order: "
        f"{[pk['index'] for pk in picks]}."
    )
    research_log = (
        "# Sequential active-learning batch design\n\n"
        "Starting from the posterior after the already-run experiments, the most "
        "informative candidate was selected, the posterior covariance was updated "
        "by the Sherman–Morrison rank-1 formula "
        "Σ' = Σ − (Σxxᵀ Σ)/(σ² + xᵀΣx), and the process repeated. Per-step EIG is "
        "non-increasing under a fixed pool (diminishing returns), and the "
        "cumulative EIG is the total expected uncertainty reduction of the batch.\n\n"
        "| Step | Candidate | EIG (nats) |\n| --- | --- | --- |\n"
        + "".join(
            f"| {pk['step']} | #{pk['index']} | {pk['expectedInformationGain']:.4g} |\n"
            for pk in picks
        )
    )
    return {
        "status": "success",
        "analysis": analysis,
        "selected": picks,
        "selectedIndices": [pk["index"] for pk in picks],
        "cumulativeEIG": round(cumulative_eig, 10),
        "nCandidates": n_cand,
        "nFeatures": d,
        "researchLog": research_log,
    }


# --------------------------------------------------------------------------- #
# Task 3 — greedy D-optimal subset selection
# --------------------------------------------------------------------------- #
def task_d_optimal_selection(p):
    try:
        import numpy as np
    except Exception as e:  # pragma: no cover - environment dependent
        _fail(f"d_optimal_selection requires numpy: {e}", status="unavailable")

    pool = p.get("candidatePool")
    if not isinstance(pool, list) or not pool:
        _fail("Provide `candidatePool` (2D array: candidates x features).")
    Xc = _load_matrix(np, pool, "candidatePool")
    if Xc.ndim != 2:
        _fail(f"`candidatePool` must be 2D; got ndim={Xc.ndim}.")
    n_cand, d = Xc.shape

    try:
        k = int(p.get("k"))
    except Exception:
        _fail("Provide integer `k` (number of experiments to select).")
    if k < 1 or k > n_cand:
        _fail(f"`k` must be between 1 and the pool size ({n_cand}).")
    try:
        noise_var = float(p.get("noiseVariance", 1.0))
        prior_precision = float(p.get("priorPrecision", 1.0))
    except Exception:
        _fail("`noiseVariance` and `priorPrecision` must be numbers.")
    if noise_var <= 0 or prior_precision <= 0:
        _fail("`noiseVariance` and `priorPrecision` must be > 0.")

    A = prior_precision * np.eye(d)
    Sigma = np.linalg.inv(A)
    remaining = list(range(n_cand))
    picks = []
    logdet0 = float(np.linalg.slogdet(A)[1])
    for _step in range(k):
        best_i = None
        best_gain = -np.inf
        best_pv = None
        for i in remaining:
            x = Xc[i]
            pv = float(x @ Sigma @ x)
            gain = float(np.log1p(pv / noise_var))  # increase in log det(A)
            if gain > best_gain:
                best_gain, best_i, best_pv = gain, i, pv
        x = Xc[best_i]
        Sx = Sigma @ x
        Sigma = Sigma - np.outer(Sx, Sx) / (noise_var + best_pv)
        picks.append({"index": best_i, "logDetGain": round(best_gain, 10)})
        remaining.remove(best_i)

    A_final = np.linalg.inv(Sigma)
    logdet_final = float(np.linalg.slogdet(A_final)[1])
    analysis = (
        f"Selected a D-optimal subset of {k} experiment(s) from {n_cand} "
        f"candidate(s): greedily maximized log det of the information matrix "
        f"(τI + XᵀX/σ²), raising it from {logdet0:.4g} to {logdet_final:.4g}. "
        f"Chosen indices: {[pk['index'] for pk in picks]}."
    )
    research_log = (
        "# Greedy D-optimal subset selection\n\n"
        "D-optimality maximizes the determinant of the information matrix "
        "(equivalently minimizes the volume of the parameter confidence "
        "ellipsoid). Starting from the prior τI, the candidate giving the largest "
        "log-det increase log(1 + xᵀΣx/σ²) was added at each step and the "
        "covariance updated by Sherman–Morrison.\n\n"
        f"| Metric | Value |\n| --- | --- |\n"
        f"| Pool size | {n_cand} |\n| Selected (k) | {k} |\n"
        f"| log det (start) | {logdet0:.4g} |\n"
        f"| log det (final) | {logdet_final:.4g} |\n"
    )
    return {
        "status": "success",
        "analysis": analysis,
        "selectedIndices": [pk["index"] for pk in picks],
        "selected": picks,
        "logDetStart": round(logdet0, 10),
        "logDetFinal": round(logdet_final, 10),
        "nCandidates": n_cand,
        "nFeatures": d,
        "researchLog": research_log,
    }


# --------------------------------------------------------------------------- #
# Task 4 — maximin Latin-Hypercube space-filling design
# --------------------------------------------------------------------------- #
def task_space_filling_design(p):
    try:
        import numpy as np
    except Exception as e:  # pragma: no cover - environment dependent
        _fail(f"space_filling_design requires numpy: {e}", status="unavailable")
    try:
        from scipy.spatial.distance import pdist
        from scipy.stats import qmc
    except Exception as e:  # pragma: no cover - environment dependent
        _fail(f"space_filling_design requires scipy: {e}", status="unavailable")

    try:
        n_points = int(p.get("nPoints"))
    except Exception:
        _fail("Provide integer `nPoints`.")
    if n_points < 2:
        _fail("`nPoints` must be >= 2.")

    bounds = p.get("bounds")
    if not isinstance(bounds, list) or not bounds:
        _fail("Provide `bounds` as [[low, high], ...] per design dimension.")
    try:
        B = np.asarray(bounds, float)
    except Exception as e:
        _fail(f"`bounds` must be numeric: {e}")
    if B.ndim != 2 or B.shape[1] != 2:
        _fail("`bounds` must be an array of [low, high] pairs.")
    if np.any(B[:, 1] <= B[:, 0]):
        _fail("Each bound must have high > low.")
    d = B.shape[0]
    seed = int(p.get("seed", 42))

    sampler = qmc.LatinHypercube(d=d, seed=seed)
    unit = sampler.random(n=n_points)  # in [0,1)^d
    scaled = qmc.scale(unit, B[:, 0], B[:, 1])

    # Space-filling quality: minimum pairwise distance (larger = better spread).
    min_dist = float(np.min(pdist(scaled))) if n_points > 1 else 0.0

    points = [[round(float(v), 8) for v in row] for row in scaled]
    analysis = (
        f"Generated a {n_points}-point Latin-Hypercube design over {d} "
        f"dimension(s) (seed={seed}); minimum pairwise distance = {min_dist:.4g}. "
        f"Each dimension is stratified into {n_points} equal-probability bins for "
        f"even coverage of the design space."
    )
    research_log = (
        "# Latin-Hypercube space-filling design\n\n"
        f"A Latin-Hypercube sample of **{n_points}** points was drawn over "
        f"**{d}** design dimension(s) and scaled to the provided bounds "
        "(`scipy.stats.qmc`). Latin-Hypercube sampling stratifies every dimension "
        "into equal-probability intervals, guaranteeing marginal coverage far more "
        "evenly than uniform random sampling — the standard initial screen before "
        "model-based (Bayesian) optimal design takes over.\n\n"
        f"| Metric | Value |\n| --- | --- |\n"
        f"| Points | {n_points} |\n| Dimensions | {d} |\n| Seed | {seed} |\n"
        f"| Min pairwise distance | {min_dist:.4g} |\n"
    )
    return {
        "status": "success",
        "analysis": analysis,
        "points": points,
        "minPairwiseDistance": round(min_dist, 10),
        "nPoints": n_points,
        "nDimensions": d,
        "seed": seed,
        "researchLog": research_log,
    }


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
TASKS = {
    "bayesian_optimal_design": task_bayesian_optimal_design,
    "sequential_active_learning": task_sequential_active_learning,
    "d_optimal_selection": task_d_optimal_selection,
    "space_filling_design": task_space_filling_design,
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
