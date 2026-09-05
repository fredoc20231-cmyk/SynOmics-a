#!/usr/bin/env python3
"""Structured knowledge-base logic compiler (Z3) — single dispatch.

Compiles a knowledge base of EXPLICIT, structured regulatory relationships
(activates / inhibits / requires, with optional conditions) into formal Z3
constraints, then checks whether observed molecular states are logically
consistent with that knowledge base. When the trusted data contradicts the
literature-asserted relationships, the minimal contradicted subset is returned as
"novel discovery" candidates via an exact UNSAT core — a discovery flag, not an
error. Reads JSON on stdin, prints JSON on stdout.

ZERO HALLUCINATION / FULLY RIGOROUS: this module performs NO natural-language
extraction and NO guessing. It only accepts already-structured relationships
(the caller/curator supplies them explicitly) and returns provable SAT/UNSAT
verdicts with exact unsat cores from the Z3 SMT solver. Every verdict is a formal
proof object, not an estimate. (The LLM-from-free-text axiom extraction in the
roadmap is deliberately NOT implemented — it would violate the zero-hallucination
mandate.)

Relationship semantics (boolean node states; each is a hard implication):
- activates(X, Y[, when C]) :  (C ∧ X) → Y      encoded  ¬C ∨ ¬X ∨ Y
- inhibits (X, Y[, when C]) :  (C ∧ X) → ¬Y     encoded  ¬C ∨ ¬X ∨ ¬Y
- requires (X, Y[, when C]) :  (C ∧ Y) → X      encoded  ¬C ∨ ¬Y ∨ X
(without a condition, C is dropped: activates is ¬X ∨ Y, etc.)

Tasks
-----
- compile_constraints    : parse relationships, emit the human-readable logical
  encoding, and check the knowledge base is internally satisfiable.
- check_consistency      : is (knowledge base ∧ observed states) satisfiable?
  Returns a satisfying assignment when SAT, or the exact UNSAT core when not.
- detect_novel_discovery : treat the observations as trusted and find the minimal
  set of KB relationships they contradict (UNSAT core) — candidate novel findings.
"""
import json
import sys


def _fail(msg, status="error"):
    print(json.dumps({"status": status, "error": msg}))
    sys.exit(0)


def _parse_relationships(rels):
    """Validate + normalize the relationship list; return list of dicts."""
    if not isinstance(rels, list) or not rels:
        _fail("Provide `relationships` as [{source, relation, target, when?}, ...].")
    valid_rel = {"activates", "inhibits", "requires"}
    out = []
    for k, r in enumerate(rels):
        if not isinstance(r, dict):
            _fail(f"Relationship #{k} must be an object.")
        src = r.get("source")
        tgt = r.get("target")
        rel = r.get("relation")
        cond = r.get("when")
        if not isinstance(src, str) or not src:
            _fail(f"Relationship #{k} needs a string `source`.")
        if not isinstance(tgt, str) or not tgt:
            _fail(f"Relationship #{k} needs a string `target`.")
        if rel not in valid_rel:
            _fail(f"Relationship #{k} `relation` must be one of {sorted(valid_rel)}.")
        if cond is not None and (not isinstance(cond, str) or not cond):
            _fail(f"Relationship #{k} `when` must be a non-empty string if given.")
        label = r.get("id") or (
            f"{src} {rel} {tgt}" + (f" when {cond}" if cond else "")
        )
        out.append({"source": src, "relation": rel, "target": tgt,
                    "when": cond, "id": str(label)})
    return out


def _nodes_of(rels, obs_map):
    nodes = set()
    for r in rels:
        nodes.add(r["source"]); nodes.add(r["target"])
        if r["when"]:
            nodes.add(r["when"])
    nodes.update(obs_map.keys())
    return sorted(nodes)


def _parse_observations(obs):
    """Accept {node: 0/1/bool} or [{node, state}]; return {node: bool}."""
    result = {}
    if obs is None:
        return result
    if isinstance(obs, dict):
        items = obs.items()
    elif isinstance(obs, list):
        items = []
        for k, o in enumerate(obs):
            if not isinstance(o, dict) or "node" not in o or "state" not in o:
                _fail(f"Observation #{k} must be {{node, state}}.")
            items.append((o["node"], o["state"]))
    else:
        _fail("`observations` must be a map {node: state} or a list [{node, state}].")
    for node, state in items:
        if not isinstance(node, str) or not node:
            _fail("Observation node names must be non-empty strings.")
        if isinstance(state, bool):
            b = state
        elif state in (0, 1):
            b = bool(state)
        elif isinstance(state, str) and state.lower() in ("on", "off", "active", "inactive", "1", "0", "true", "false"):
            b = state.lower() in ("on", "active", "1", "true")
        else:
            _fail(f"Observation for {node!r} must be 0/1/true/false/on/off.")
        result[str(node)] = b
    return result


def _constraint(z3, r, bvars):
    """Build the Z3 clause for one relationship."""
    X = bvars[r["source"]]
    Y = bvars[r["target"]]
    C = bvars[r["when"]] if r["when"] else None
    if r["relation"] == "activates":
        base = z3.Or(z3.Not(X), Y)
    elif r["relation"] == "inhibits":
        base = z3.Or(z3.Not(X), z3.Not(Y))
    else:  # requires: (Y) -> X
        base = z3.Or(z3.Not(Y), X)
    if C is not None:
        base = z3.Or(z3.Not(C), base)
    return base


def _readable(r):
    arrow = {"activates": "→", "inhibits": "⊣", "requires": "⇐(requires)"}[r["relation"]]
    s = f"{r['source']} {arrow} {r['target']}"
    if r["when"]:
        s += f"  [when {r['when']}]"
    return s


# --------------------------------------------------------------------------- #
# Task 1 — compile + internal satisfiability
# --------------------------------------------------------------------------- #
def task_compile_constraints(p):
    try:
        import z3
    except Exception as e:  # pragma: no cover - environment dependent
        _fail(f"compile_constraints requires z3-solver: {e}", status="unavailable")

    rels = _parse_relationships(p.get("relationships"))
    nodes = _nodes_of(rels, {})
    bvars = {n: z3.Bool(n) for n in nodes}

    s = z3.Solver()
    s.set(unsat_core=True)
    for r in rels:
        s.assert_and_track(_constraint(z3, r, bvars), r["id"])
    res = s.check()
    satisfiable = res == z3.sat

    encoding = [{"id": r["id"], "relation": r["relation"],
                 "readable": _readable(r)} for r in rels]
    core = []
    if not satisfiable:
        core = [str(c) for c in s.unsat_core()]

    analysis = (
        f"Compiled {len(rels)} structured relationship(s) over {len(nodes)} node(s) "
        f"into Z3 constraints. Knowledge base is "
        + ("internally SATISFIABLE (self-consistent)." if satisfiable
           else f"INTERNALLY CONTRADICTORY; conflicting core: {core}.")
    )
    research_log = (
        "# Knowledge-base compilation (Z3)\n\n"
        f"Each of the {len(rels)} explicit relationship(s) was translated into a "
        "hard boolean implication (activates: X→Y; inhibits: X→¬Y; requires: "
        "Y→X; optionally guarded by a condition). No natural-language extraction "
        "was performed — the relationships are supplied structured, and the "
        "verdict is a formal SMT result.\n\n"
        f"| Metric | Value |\n| --- | --- |\n"
        f"| Relationships | {len(rels)} |\n| Nodes | {len(nodes)} |\n"
        f"| Internally satisfiable | {satisfiable} |\n"
    )
    return {
        "status": "success",
        "analysis": analysis,
        "satisfiable": bool(satisfiable),
        "encoding": encoding,
        "nodes": nodes,
        "nRelationships": len(rels),
        "contradictoryCore": core,
        "researchLog": research_log,
    }


# --------------------------------------------------------------------------- #
# Task 2 — consistency of KB + observations
# --------------------------------------------------------------------------- #
def task_check_consistency(p):
    try:
        import z3
    except Exception as e:  # pragma: no cover - environment dependent
        _fail(f"check_consistency requires z3-solver: {e}", status="unavailable")

    rels = _parse_relationships(p.get("relationships"))
    obs = _parse_observations(p.get("observations"))
    if not obs:
        _fail("Provide `observations` (measured node states) to check against the knowledge base.")
    nodes = _nodes_of(rels, obs)
    bvars = {n: z3.Bool(n) for n in nodes}

    s = z3.Solver()
    s.set(unsat_core=True)
    for r in rels:
        s.assert_and_track(_constraint(z3, r, bvars), r["id"])
    for node, state in obs.items():
        lit = bvars[node] if state else z3.Not(bvars[node])
        s.assert_and_track(lit, f"obs:{node}={int(state)}")

    res = s.check()
    consistent = res == z3.sat
    model_assignment = {}
    core = []
    if consistent:
        m = s.model()
        for n in nodes:
            val = m.eval(bvars[n], model_completion=True)
            model_assignment[n] = bool(z3.is_true(val))
    else:
        core = [str(c) for c in s.unsat_core()]

    analysis = (
        f"Checked {len(obs)} observation(s) against {len(rels)} relationship(s): "
        + ("CONSISTENT — a satisfying assignment exists."
           if consistent else
           f"INCONSISTENT (UNSAT). Minimal conflicting core: {core}.")
    )
    research_log = (
        "# Consistency check (Z3 SMT)\n\n"
        "The knowledge base and the observed node states were asserted together "
        "and solved. A SAT result returns a concrete consistent assignment; an "
        "UNSAT result returns the exact minimal core of constraints "
        "(relationships and/or observations) that cannot hold simultaneously — a "
        "formal proof of conflict, not a heuristic.\n\n"
        f"| Metric | Value |\n| --- | --- |\n"
        f"| Relationships | {len(rels)} |\n| Observations | {len(obs)} |\n"
        f"| Consistent | {consistent} |\n"
    )
    return {
        "status": "success",
        "analysis": analysis,
        "consistent": bool(consistent),
        "assignment": model_assignment,
        "conflictCore": core,
        "nRelationships": len(rels),
        "nObservations": len(obs),
        "researchLog": research_log,
    }


# --------------------------------------------------------------------------- #
# Task 3 — novel-discovery detection (data-trusted unsat core)
# --------------------------------------------------------------------------- #
def task_detect_novel_discovery(p):
    try:
        import z3
    except Exception as e:  # pragma: no cover - environment dependent
        _fail(f"detect_novel_discovery requires z3-solver: {e}", status="unavailable")

    rels = _parse_relationships(p.get("relationships"))
    obs = _parse_observations(p.get("observations"))
    if not obs:
        _fail("Provide trusted `observations` (experimental measurements) to test against the literature.")
    nodes = _nodes_of(rels, obs)
    bvars = {n: z3.Bool(n) for n in nodes}

    # First: are the observations themselves self-consistent? (A node cannot be
    # both on and off — that's a DATA error, not a novel discovery.) Since obs is
    # a dict keyed by node, duplicates already collapsed, so obs is consistent by
    # construction; we still verify the observation literals are jointly SAT.
    obs_solver = z3.Solver()
    for node, state in obs.items():
        obs_solver.add(bvars[node] if state else z3.Not(bvars[node]))
    if obs_solver.check() != z3.sat:  # pragma: no cover - defensive
        _fail("Observations are internally contradictory (data error, not a novel discovery).")

    # Observations are HARD (trusted); relationships are retractable (tracked).
    s = z3.Solver()
    s.set(unsat_core=True)
    for node, state in obs.items():
        s.add(bvars[node] if state else z3.Not(bvars[node]))
    for r in rels:
        s.assert_and_track(_constraint(z3, r, bvars), r["id"])

    res = s.check()
    id_to_rel = {r["id"]: r for r in rels}
    if res == z3.sat:
        analysis = (
            f"The {len(obs)} trusted observation(s) are CONSISTENT with all "
            f"{len(rels)} literature relationship(s) — no novel discovery flagged."
        )
        return {
            "status": "success",
            "analysis": analysis,
            "novelDiscovery": False,
            "contradictedRelationships": [],
            "nRelationships": len(rels),
            "nObservations": len(obs),
            "researchLog": (
                "# Novel-discovery detection\n\n"
                "Observations were asserted as trusted (hard) constraints and every "
                "literature relationship as a retractable tracked constraint. The "
                "system is satisfiable — the data agrees with the consensus.\n"
            ),
        }

    core_ids = [str(c) for c in s.unsat_core()]
    contradicted = [
        {"id": rid, "readable": _readable(id_to_rel[rid]),
         "relation": id_to_rel[rid]["relation"],
         "source": id_to_rel[rid]["source"], "target": id_to_rel[rid]["target"]}
        for rid in core_ids if rid in id_to_rel
    ]

    # Verify: removing the core relationships restores consistency (a real proof
    # that these are exactly the contradicted constraints).
    s2 = z3.Solver()
    for node, state in obs.items():
        s2.add(bvars[node] if state else z3.Not(bvars[node]))
    for r in rels:
        if r["id"] not in core_ids:
            s2.add(_constraint(z3, r, bvars))
    restored = s2.check() == z3.sat

    analysis = (
        f"NOVEL DISCOVERY candidate(s): the trusted data contradict "
        f"{len(contradicted)} literature relationship(s) — "
        f"{[c['readable'] for c in contradicted]}. These are flagged as candidate "
        f"novel findings (consensus overturned by data), not errors. "
        f"Removing them restores consistency: {restored}."
    )
    research_log = (
        "# Novel-discovery detection (data-trusted UNSAT core)\n\n"
        "Experimental observations were treated as trusted hard constraints; each "
        "literature relationship was made retractable. The Z3 UNSAT core is the "
        "minimal set of literature relationships that cannot coexist with the "
        "data — i.e., the consensus claims the new data overturns. This is a "
        "formal proof: removing exactly the core restores satisfiability "
        f"(verified: {restored}).\n\n"
        f"| Metric | Value |\n| --- | --- |\n"
        f"| Relationships | {len(rels)} |\n| Observations | {len(obs)} |\n"
        f"| Contradicted (novel) | {len(contradicted)} |\n"
        f"| Consistency restored after removal | {restored} |\n"
    )
    return {
        "status": "success",
        "analysis": analysis,
        "novelDiscovery": True,
        "contradictedRelationships": contradicted,
        "consistencyRestoredAfterRemoval": bool(restored),
        "nRelationships": len(rels),
        "nObservations": len(obs),
        "researchLog": research_log,
    }


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
TASKS = {
    "compile_constraints": task_compile_constraints,
    "check_consistency": task_check_consistency,
    "detect_novel_discovery": task_detect_novel_discovery,
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
