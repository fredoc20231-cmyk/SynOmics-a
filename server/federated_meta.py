#!/usr/bin/env python3
"""Federated meta-analysis — privacy-preserving cross-site statistics (dispatch).

The honest core of a "federated swarm": compute combined statistical
significance across datasets held at different institutions using ONLY per-site
summary/sufficient statistics — raw records never leave a site. Reads JSON on
stdin, prints JSON on stdout.

ZERO HALLUCINATION: because (n, mean, variance) are *sufficient statistics* for a
Gaussian, and z-scores / p-values combine by exact formulae, the federated
result is mathematically identical to the analysis on the pooled raw data — it
is a real computation, not an approximation, and nothing is invented. No site's
individual rows are required or reconstructed.

Tasks
-----
- federated_ttest      : pooled two-group Welch t-test from per-site
  (n, mean, variance) sufficient statistics for groups A and B. Exactly equals
  the t-test on the concatenated raw data.
- stouffer_meta        : combine per-site z-scores (optionally weighted) into one
  z + p (Stouffer's method).
- fisher_meta          : combine per-site p-values via Fisher's method
  (−2·Σ ln p ~ χ²₂ₖ).
- random_effects_meta  : DerSimonian–Laird random-effects pooling of per-site
  effect sizes + standard errors → pooled effect, Q, I², τ².
"""
import json
import math
import sys


def _fail(msg, status="error"):
    print(json.dumps({"status": status, "error": msg}))
    sys.exit(0)


# --------------------------------------------------------------------------- #
# Task 1 — federated two-group Welch t-test from sufficient statistics
# --------------------------------------------------------------------------- #
def _pool_group(np, ns, means, vars):
    """Exact pooled (N, mean, variance) from per-site (n, mean, var).

    Uses additive sums of squares: SS_total = Σ[(n_i−1)·var_i + n_i·mean_i²] −
    N·grand_mean², giving the pooled sample variance SS_total/(N−1).
    """
    ns = np.asarray(ns, float)
    means = np.asarray(means, float)
    vars = np.asarray(vars, float)
    N = float(np.sum(ns))
    grand_mean = float(np.sum(ns * means) / N)
    ss = float(np.sum((ns - 1.0) * vars + ns * means**2) - N * grand_mean**2)
    var = ss / (N - 1.0) if N > 1 else float("nan")
    return N, grand_mean, var


def task_federated_ttest(p):
    try:
        import numpy as np
        from scipy import stats
    except Exception as e:  # pragma: no cover - environment dependent
        _fail(f"federated_ttest requires numpy+scipy: {e}", status="unavailable")

    sites = p.get("sites")
    if not isinstance(sites, list) or not sites:
        _fail(
            "Provide `sites` as [{nA, meanA, varA, nB, meanB, varB}, ...] — per-site "
            "sufficient statistics for the two groups (no raw rows)."
        )
    nA, mA, vA, nB, mB, vB = [], [], [], [], [], []
    for k, s in enumerate(sites):
        try:
            nA.append(float(s["nA"])); mA.append(float(s["meanA"])); vA.append(float(s["varA"]))
            nB.append(float(s["nB"])); mB.append(float(s["meanB"])); vB.append(float(s["varB"]))
        except Exception:
            _fail(f"Site #{k} must have numeric nA, meanA, varA, nB, meanB, varB.")
    if any(n < 2 for n in nA) or any(n < 2 for n in nB):
        _fail("Every site needs nA>=2 and nB>=2 (a variance requires >=2 samples).")
    if any(v < 0 for v in vA + vB):
        _fail("Variances must be non-negative.")

    NA, meanA, varA = _pool_group(np, nA, mA, vA)
    NB, meanB, varB = _pool_group(np, nB, mB, vB)

    # Welch t-test on the pooled sufficient statistics.
    seA2 = varA / NA
    seB2 = varB / NB
    se = math.sqrt(seA2 + seB2)
    if se == 0:
        _fail("Zero pooled standard error (both groups constant); t is undefined.")
    t = (meanA - meanB) / se
    # Welch–Satterthwaite degrees of freedom.
    df = (seA2 + seB2) ** 2 / (seA2**2 / (NA - 1) + seB2**2 / (NB - 1))
    pval = float(2.0 * stats.t.sf(abs(t), df))
    # Cohen's d with pooled SD (using the pooled within-group variance).
    pooled_sd = math.sqrt(((NA - 1) * varA + (NB - 1) * varB) / (NA + NB - 2))
    cohen_d = (meanA - meanB) / pooled_sd if pooled_sd > 0 else float("nan")

    analysis = (
        f"Federated Welch t-test across {len(sites)} site(s) using only "
        f"sufficient statistics (no raw rows): pooled group A (N={int(NA)}, "
        f"mean={meanA:.4g}) vs group B (N={int(NB)}, mean={meanB:.4g}); "
        f"t={t:.4g}, df={df:.3g}, p={pval:.4g}, Cohen's d={cohen_d:.3g}. This "
        f"equals the t-test on the pooled raw data exactly."
    )
    research_log = (
        "# Federated two-group Welch t-test\n\n"
        "Each site contributed only (n, mean, variance) per group — never raw "
        "records. Because these are sufficient statistics for a Gaussian, the "
        "pooled group N, mean and variance were reconstructed exactly via additive "
        "sums of squares, and a Welch (unequal-variance) t-test computed on them. "
        "The result is identical to analysing the concatenated raw data.\n\n"
        f"| Metric | Value |\n| --- | --- |\n"
        f"| Sites | {len(sites)} |\n| N (A) | {int(NA)} |\n| N (B) | {int(NB)} |\n"
        f"| mean A | {meanA:.6g} |\n| mean B | {meanB:.6g} |\n"
        f"| t | {t:.6g} |\n| df | {df:.6g} |\n| p-value | {pval:.6g} |\n"
        f"| Cohen's d | {cohen_d:.6g} |\n"
    )
    return {
        "status": "success",
        "analysis": analysis,
        "t": round(float(t), 10),
        "df": round(float(df), 10),
        "pValue": round(pval, 12),
        "cohenD": round(float(cohen_d), 10),
        "pooledA": {"n": int(NA), "mean": round(meanA, 10), "variance": round(varA, 10)},
        "pooledB": {"n": int(NB), "mean": round(meanB, 10), "variance": round(varB, 10)},
        "nSites": len(sites),
        "researchLog": research_log,
    }


# --------------------------------------------------------------------------- #
# Task 2 — Stouffer combination of per-site z-scores
# --------------------------------------------------------------------------- #
def task_stouffer_meta(p):
    try:
        import numpy as np
        from scipy import stats
    except Exception as e:  # pragma: no cover - environment dependent
        _fail(f"stouffer_meta requires numpy+scipy: {e}", status="unavailable")

    zs = p.get("zScores")
    if not isinstance(zs, list) or not zs:
        _fail("Provide `zScores` (per-site z-scores).")
    try:
        z = np.asarray(zs, float)
    except Exception as e:
        _fail(f"`zScores` must be numeric: {e}")
    if not np.all(np.isfinite(z)):
        _fail("`zScores` must be finite.")

    weights = p.get("weights")
    if weights is not None:
        try:
            w = np.asarray(weights, float)
        except Exception as e:
            _fail(f"`weights` must be numeric: {e}")
        if w.shape != z.shape:
            _fail("`weights` must match `zScores` in length.")
        if np.any(w <= 0):
            _fail("`weights` must be positive.")
    else:
        w = np.ones_like(z)

    combined = float(np.sum(w * z) / math.sqrt(float(np.sum(w**2))))
    pval = float(2.0 * stats.norm.sf(abs(combined)))
    analysis = (
        f"Stouffer combination of {z.size} site z-score(s) "
        f"({'weighted' if weights is not None else 'unweighted'}): combined "
        f"Z={combined:.4g}, two-sided p={pval:.4g}."
    )
    research_log = (
        "# Stouffer's Z combination\n\n"
        "Per-site z-scores were combined as Z = Σ wᵢzᵢ / √(Σ wᵢ²) — the "
        "weighted Stouffer method — and converted to a two-sided p-value. Only "
        "per-site z-scores are shared, never raw data.\n\n"
        f"| Metric | Value |\n| --- | --- |\n| Sites | {z.size} |\n"
        f"| Combined Z | {combined:.6g} |\n| p-value | {pval:.6g} |\n"
    )
    return {
        "status": "success",
        "analysis": analysis,
        "combinedZ": round(combined, 10),
        "pValue": round(pval, 12),
        "nSites": int(z.size),
        "researchLog": research_log,
    }


# --------------------------------------------------------------------------- #
# Task 3 — Fisher combination of per-site p-values
# --------------------------------------------------------------------------- #
def task_fisher_meta(p):
    try:
        import numpy as np
        from scipy import stats
    except Exception as e:  # pragma: no cover - environment dependent
        _fail(f"fisher_meta requires numpy+scipy: {e}", status="unavailable")

    ps = p.get("pValues")
    if not isinstance(ps, list) or not ps:
        _fail("Provide `pValues` (per-site p-values in (0, 1]).")
    try:
        pv = np.asarray(ps, float)
    except Exception as e:
        _fail(f"`pValues` must be numeric: {e}")
    if np.any(pv <= 0) or np.any(pv > 1):
        _fail("`pValues` must be in the interval (0, 1].")

    stat = float(-2.0 * np.sum(np.log(pv)))
    dof = 2 * pv.size
    combined_p = float(stats.chi2.sf(stat, dof))
    analysis = (
        f"Fisher combination of {pv.size} site p-value(s): χ²={stat:.4g} on "
        f"{dof} df → combined p={combined_p:.4g}."
    )
    research_log = (
        "# Fisher's method (p-value combination)\n\n"
        "Per-site p-values were combined as X = −2·Σ ln(pᵢ), which follows a "
        "χ² distribution with 2k degrees of freedom under the joint null. Only "
        "per-site p-values are shared.\n\n"
        f"| Metric | Value |\n| --- | --- |\n| Sites (k) | {pv.size} |\n"
        f"| χ² statistic | {stat:.6g} |\n| df | {dof} |\n"
        f"| combined p | {combined_p:.6g} |\n"
    )
    return {
        "status": "success",
        "analysis": analysis,
        "chiSquare": round(stat, 10),
        "degreesOfFreedom": dof,
        "combinedPValue": round(combined_p, 12),
        "nSites": int(pv.size),
        "researchLog": research_log,
    }


# --------------------------------------------------------------------------- #
# Task 4 — DerSimonian–Laird random-effects meta-analysis
# --------------------------------------------------------------------------- #
def task_random_effects_meta(p):
    try:
        import numpy as np
        from scipy import stats
    except Exception as e:  # pragma: no cover - environment dependent
        _fail(f"random_effects_meta requires numpy+scipy: {e}", status="unavailable")

    effects = p.get("effects")
    ses = p.get("standardErrors")
    if not isinstance(effects, list) or not effects:
        _fail("Provide `effects` (per-site effect sizes).")
    if not isinstance(ses, list) or not ses:
        _fail("Provide `standardErrors` (per-site standard errors).")
    try:
        y = np.asarray(effects, float)
        se = np.asarray(ses, float)
    except Exception as e:
        _fail(f"`effects`/`standardErrors` must be numeric: {e}")
    if y.shape != se.shape:
        _fail("`effects` and `standardErrors` must have the same length.")
    if np.any(se <= 0):
        _fail("`standardErrors` must be positive.")
    k = y.size
    if k < 2:
        _fail("Need >=2 sites for meta-analysis.")

    v = se**2
    w = 1.0 / v  # fixed-effect weights
    ybar_fixed = float(np.sum(w * y) / np.sum(w))
    Q = float(np.sum(w * (y - ybar_fixed) ** 2))
    c = float(np.sum(w) - np.sum(w**2) / np.sum(w))
    tau2 = max(0.0, (Q - (k - 1)) / c) if c > 0 else 0.0
    i2 = max(0.0, (Q - (k - 1)) / Q) * 100.0 if Q > 0 else 0.0

    w_re = 1.0 / (v + tau2)
    pooled = float(np.sum(w_re * y) / np.sum(w_re))
    pooled_se = math.sqrt(1.0 / float(np.sum(w_re)))
    z = pooled / pooled_se
    pval = float(2.0 * stats.norm.sf(abs(z)))
    ci_low = pooled - 1.959963984540054 * pooled_se
    ci_high = pooled + 1.959963984540054 * pooled_se
    q_p = float(stats.chi2.sf(Q, k - 1))

    analysis = (
        f"Random-effects (DerSimonian–Laird) meta-analysis of {k} site(s): "
        f"pooled effect={pooled:.4g} (95% CI {ci_low:.4g}..{ci_high:.4g}), "
        f"z={z:.3g}, p={pval:.4g}; heterogeneity Q={Q:.3g} (p={q_p:.3g}), "
        f"I²={i2:.3g}%, τ²={tau2:.4g}."
    )
    research_log = (
        "# Random-effects meta-analysis (DerSimonian–Laird)\n\n"
        "Per-site effect sizes and standard errors were pooled with "
        "inverse-variance weights augmented by the between-site variance τ² "
        "(DerSimonian–Laird estimator). Cochran's Q and I² quantify "
        "heterogeneity. Only per-site summary effects are shared, never raw "
        "data.\n\n"
        f"| Metric | Value |\n| --- | --- |\n| Sites | {k} |\n"
        f"| Pooled effect | {pooled:.6g} |\n"
        f"| 95% CI | [{ci_low:.6g}, {ci_high:.6g}] |\n"
        f"| z | {z:.6g} |\n| p-value | {pval:.6g} |\n"
        f"| Q | {Q:.6g} |\n| I² (%) | {i2:.6g} |\n| τ² | {tau2:.6g} |\n"
    )
    return {
        "status": "success",
        "analysis": analysis,
        "pooledEffect": round(pooled, 10),
        "pooledSE": round(pooled_se, 10),
        "ci95": [round(ci_low, 10), round(ci_high, 10)],
        "z": round(float(z), 10),
        "pValue": round(pval, 12),
        "Q": round(Q, 10),
        "QpValue": round(q_p, 12),
        "I2Percent": round(i2, 6),
        "tau2": round(tau2, 10),
        "nSites": int(k),
        "researchLog": research_log,
    }


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
TASKS = {
    "federated_ttest": task_federated_ttest,
    "stouffer_meta": task_stouffer_meta,
    "fisher_meta": task_fisher_meta,
    "random_effects_meta": task_random_effects_meta,
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
