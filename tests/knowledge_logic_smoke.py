#!/usr/bin/env python3
"""Knowledge-logic gate — every verdict is a checked Z3 proof.

A self-consistent KB compiles SAT; observations that violate an implication are
proven INCONSISTENT with an exact conflict core; trusted data that contradicts a
literature relationship yields the minimal contradicted set (novel discovery),
and removing exactly that set is proven to restore consistency. Conditions gate
rules correctly."""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "server", "knowledge_logic.py")

try:
    import z3  # noqa: F401
except Exception as e:  # noqa: BLE001
    print(f"SKIP: z3 not available ({e}).")
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


KB = [
    {"source": "A", "relation": "activates", "target": "B"},
    {"source": "B", "relation": "activates", "target": "C"},
]

# ---- compile_constraints ----
c = run({"task": "compile_constraints", "relationships": KB})
check("compile status success", c["status"] == "success", c)
check("compile: KB internally satisfiable", c["satisfiable"] is True, c)
check("compile lists nodes A,B,C", set(c["nodes"]) == {"A", "B", "C"}, c)
check("compile encodes 2 relationships", c["nRelationships"] == 2, c)

# ---- check_consistency ----
# A=1, B=1, C=1 is consistent with A→B→C.
ok = run({"task": "check_consistency", "relationships": KB,
          "observations": {"A": 1, "B": 1, "C": 1}})
check("consistency: satisfying observation is consistent", ok["consistent"] is True, ok)
check("consistency returns an assignment", ok["assignment"].get("C") is True, ok)

# A=1, B=0 violates A activates B -> UNSAT with a core naming that relationship + obs.
bad = run({"task": "check_consistency", "relationships": KB,
           "observations": {"A": 1, "B": 0}})
check("consistency: A=1,B=0 is INCONSISTENT", bad["consistent"] is False, bad)
core_join = " ".join(bad["conflictCore"])
check("consistency core references A activates B", "A activates B" in core_join, bad["conflictCore"])
check("consistency core references the B observation", "obs:B=0" in core_join, bad["conflictCore"])

# ---- detect_novel_discovery ----
# Trusted data A=1,B=1,C=0 contradicts exactly 'B activates C'.
nd = run({"task": "detect_novel_discovery", "relationships": KB,
          "observations": {"A": 1, "B": 1, "C": 0}})
check("novel status success", nd["status"] == "success", nd)
check("novel discovery flagged", nd["novelDiscovery"] is True, nd)
ids = [r["id"] for r in nd["contradictedRelationships"]]
check("novel: contradicted set is exactly {B activates C}", ids == ["B activates C"], nd)
check("novel: removing the core restores consistency (proven)",
      nd["consistencyRestoredAfterRemoval"] is True, nd)

# Data consistent with the KB -> no novel discovery.
nd_none = run({"task": "detect_novel_discovery", "relationships": KB,
               "observations": {"A": 0, "B": 0, "C": 0}})
check("novel: consistent data flags nothing", nd_none["novelDiscovery"] is False, nd_none)

# ---- inhibits semantics ----
inh = [{"source": "R", "relation": "inhibits", "target": "S"}]
inh_bad = run({"task": "check_consistency", "relationships": inh,
               "observations": {"R": 1, "S": 1}})
check("inhibits: R=1,S=1 is inconsistent", inh_bad["consistent"] is False, inh_bad)
inh_ok = run({"task": "check_consistency", "relationships": inh,
              "observations": {"R": 1, "S": 0}})
check("inhibits: R=1,S=0 is consistent", inh_ok["consistent"] is True, inh_ok)

# ---- conditional rule gating ----
cond = [{"source": "A", "relation": "activates", "target": "B", "when": "C"}]
# C off -> rule inactive -> A=1,B=0 consistent.
cond_off = run({"task": "check_consistency", "relationships": cond,
                "observations": {"A": 1, "B": 0, "C": 0}})
check("condition off -> rule inactive -> consistent", cond_off["consistent"] is True, cond_off)
# C on -> rule active -> A=1,B=0 inconsistent.
cond_on = run({"task": "check_consistency", "relationships": cond,
               "observations": {"A": 1, "B": 0, "C": 1}})
check("condition on -> rule active -> inconsistent", cond_on["consistent"] is False, cond_on)

# ---- requires semantics ----
req = [{"source": "X", "relation": "requires", "target": "Y"}]  # Y needs X: Y->X
req_bad = run({"task": "check_consistency", "relationships": req,
               "observations": {"Y": 1, "X": 0}})
check("requires: Y=1,X=0 is inconsistent", req_bad["consistent"] is False, req_bad)

# ---- error handling ----
check("unknown task -> error", run({"task": "nope"}).get("status") == "error")
check("missing relationships -> error",
      run({"task": "compile_constraints"}).get("status") == "error")
check("bad relation verb -> error",
      run({"task": "compile_constraints",
           "relationships": [{"source": "A", "relation": "frobnicates", "target": "B"}]}).get("status") == "error")
check("check_consistency without observations -> error",
      run({"task": "check_consistency", "relationships": KB}).get("status") == "error")

print(f"\nALL {passed} KNOWLEDGE-LOGIC TESTS PASSED")
