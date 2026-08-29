"""
run_transpile_study.py
----------------------
The H1 instrument. Transpiles Grover circuits onto a fixed heavy-hex coupling
map and measures how the two-qubit gate cost of ONE Grover iteration grows
with n.

Method
  1. For each n, transpile the circuit at several values of k.
  2. Fit  total_2q_gates(k) = intercept + slope * k.  The slope is the marginal
     two-qubit cost of one Grover iteration, cleanly separated from state
     preparation and measurement.
  3. Fit  log(slope) = log(a) + b * log(n).  The exponent b is the H1 statistic:
     b <= 1 is linear-or-better, b > 1 is super-linear.

No quantum hardware is needed. Transpilation is a compile-time transformation,
so this whole study runs on a laptop and is unaffected by device queues. That
is why H1 survives the week-three cut rule.

Run:   python run_transpile_study.py
Out:   results/transpile_study.csv  and  figures/h1_gate_scaling.png
"""

from __future__ import annotations

import os
import csv

import numpy as np
from scipy import stats

from qiskit import transpile
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

from grover_core import grover_circuit

# ---------------------------------------------------------------- config ----
N_RANGE   = range(3, 11)      # raise the ceiling once you know the timing.
                              # n=2 is EXCLUDED on purpose: Grover on N=4 is
                              # exact and periodic in k, so the transpiler
                              # cancels whole iterations away and the
                              # per-iteration slope comes out negative. Worth a
                              # sentence in the methods section, not a bug.
K_PROBES  = (1, 2, 3)         # k values used to extract the per-iteration slope
OPT_LEVEL = 2                 # FIXED across all comparisons. Log this choice.
SEED      = 20260815
MARKED    = 0                 # placement does not change gate count materially;
                              # vary it if you want to confirm that empirically
TWO_QUBIT_NAMES = {"cz", "cx", "ecr", "swap", "rzz"}
# ----------------------------------------------------------------------------


def two_qubit_count(circuit) -> int:
    ops = circuit.count_ops()
    return sum(v for k, v in ops.items() if k in TWO_QUBIT_NAMES)


def main() -> None:
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    backend = FakeSherbrooke()          # IBM heavy-hex, 127 qubits
    print(f"backend: {backend.name}   basis: {backend.operation_names}")
    print(f"{'n':>3} {'N':>7} " + " ".join(f"{'2q(k='+str(k)+')':>10}" for k in K_PROBES)
          + f" {'per-iter':>9} {'depth(k=1)':>11}")

    rows = []
    for n in N_RANGE:
        counts, depth1 = [], None
        for k in K_PROBES:
            qc = grover_circuit(n, MARKED, k)
            tq = transpile(qc, backend, optimization_level=OPT_LEVEL,
                           seed_transpiler=SEED)
            counts.append(two_qubit_count(tq))
            if k == 1:
                depth1 = tq.depth()

        slope, intercept, *_ = stats.linregress(K_PROBES, counts)
        rows.append(dict(n=n, N=2 ** n,
                         **{f"two_qubit_k{k}": c for k, c in zip(K_PROBES, counts)},
                         per_iteration_2q=slope, transpiled_depth_k1=depth1))
        print(f"{n:>3} {2**n:>7} " + " ".join(f"{c:>10}" for c in counts)
              + f" {slope:>9.1f} {depth1:>11}")

    # ---- power-law fit: per-iteration cost ~ a * n**b -----------------------
    ns = np.array([r["n"] for r in rows], dtype=float)
    gates = np.array([r["per_iteration_2q"] for r in rows], dtype=float)
    fit = stats.linregress(np.log(ns), np.log(gates))

    dof = len(ns) - 2
    t_crit = stats.t.ppf(0.975, dof)
    b, se = fit.slope, fit.stderr
    lo, hi = b - t_crit * se, b + t_crit * se

    print(f"\nPower-law fit  per-iteration 2q gates = {np.exp(fit.intercept):.2f} * n^b")
    print(f"  b   = {b:.3f}")
    print(f"  95% CI = [{lo:.3f}, {hi:.3f}]     R^2 = {fit.rvalue**2:.4f}")
    if lo > 1.0:
        print("  -> CI excludes 1.0 from above: SUPER-LINEAR. H1 not refuted.")
    elif hi <= 1.0:
        print("  -> CI at or below 1.0: H1 REFUTED by its own stated criterion.")
    else:
        print("  -> CI straddles 1.0: inconclusive. Widen N_RANGE.")

    with open("results/transpile_study.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    plot(ns, gates, fit, (lo, hi))
    print("\nWrote results/transpile_study.csv and figures/h1_gate_scaling.png")


def plot(ns, gates, fit, ci) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OKABE_ITO = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9",
                 "green": "#009E73", "blue": "#0072B2", "vermillion": "#D55E00"}

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(ns, gates, color=OKABE_ITO["blue"], linestyle="none", marker="o",
            markersize=7, label="measured (transpiled)")

    grid = np.linspace(ns.min(), ns.max(), 200)
    ax.plot(grid, np.exp(fit.intercept) * grid ** fit.slope,
            color=OKABE_ITO["vermillion"], linestyle="--", linewidth=1.8,
            label=fr"fit $\propto n^{{{fit.slope:.2f}}}$")
    ax.plot(grid, gates[0] * (grid / ns[0]), color=OKABE_ITO["black"],
            linestyle=":", linewidth=1.5, label="linear reference $n^{1}$")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("qubits $n$")
    ax.set_ylabel("two-qubit gates per Grover iteration")
    ax.set_title("H1: transpiled two-qubit cost per iteration (heavy-hex)")
    ax.grid(alpha=0.25, which="both", linewidth=0.5)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig("figures/h1_gate_scaling.png", dpi=200)


if __name__ == "__main__":
    main()