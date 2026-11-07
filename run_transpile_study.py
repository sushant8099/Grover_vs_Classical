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

Comparison arms (added in revision). The study is repeated for every topology
in TOPOLOGIES and every ancilla count in ANCILLA_ARMS: heavy-hex (FakeSherbrooke,
the main study), all-to-all (no coupling map, so no routing) and a square
lattice (FakeNighthawk); with 0 or 1 idle helper qubits. Each arm also gets an
exponential fit for comparison (AIC) and local exponents between consecutive n.
Per-arm summaries are written to results/transpile_exponents.csv.

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
from qiskit_ibm_runtime.fake_provider import FakeNighthawk, FakeSherbrooke

from grover_core import grover_circuit

# ---------------------------------------------------------------- config ----
N_RANGE   = range(3, 11)      # raise the ceiling once the timing is known.
                              # n=2 is EXCLUDED on purpose: Grover on N = 4 is exact and
                              # periodic in k, so the transpiler cancels whole iterations and
                              # the per-iteration slope comes out negative (paper, section 3.2). 
K_PROBES  = (1, 2, 3)         # k values used to extract the per-iteration slope
OPT_LEVEL = 2                 # FIXED across all comparisons.
SEED      = 20260815
MARKED    = 0                 # placement does not change gate count materially;
                              # vary it if you want to confirm that empirically
TWO_QUBIT_NAMES = {"cz", "cx", "ecr", "swap", "rzz"}
TOPOLOGIES = ("heavy-hex", "all-to-all", "square")
                              # heavy-hex : FakeSherbrooke, 127 qubits (main study)
                              # all-to-all: no coupling map, so no routing at all
                              # square    : FakeNighthawk, 120-qubit square lattice
ANCILLA_ARMS = (0, 1)         # idle helper qubits added to the circuit;
                              # 0 reproduces the original study exactly
ALL_TO_ALL_BASIS = ["ecr", "id", "rz", "sx", "x"]   # Sherbrooke's native gate set
# ----------------------------------------------------------------------------


def two_qubit_count(circuit) -> int:
    ops = circuit.count_ops()
    return sum(v for k, v in ops.items() if k in TWO_QUBIT_NAMES)


def transpile_target(topology: str) -> dict:
    """Keyword arguments for transpile() that select the topology."""
    if topology == "heavy-hex":
        return {"backend": FakeSherbrooke()}         # IBM heavy-hex, 127 qubits
    if topology == "square":
        return {"backend": FakeNighthawk()}          # IBM square lattice, 120 qubits
    if topology == "all-to-all":
        return {"basis_gates": ALL_TO_ALL_BASIS}     # no coupling map: no routing
    raise ValueError(f"unknown topology: {topology}")


def aic(residuals) -> float:
    """Akaike information criterion of a 2-parameter least-squares fit.
    Lower is better; a difference above about 10 is decisive."""
    m = len(residuals)
    return m * np.log(np.sum(np.square(residuals)) / m) + 2 * 2


def run_arm(topology: str, ancillas: int):
    """One arm of the study: every n in N_RANGE for one topology and one
    ancilla count. Returns the rows, the n values, the per-iteration costs,
    the power-law fit, and a one-row summary of the arm."""
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    target = transpile_target(topology)
    print(f"\n=== topology: {topology}   ancillas: {ancillas} ===")
    print(f"{'n':>3} {'N':>7} " + " ".join(f"{'2q(k='+str(k)+')':>10}" for k in K_PROBES)
          + f" {'per-iter':>9} {'depth(k=1)':>11}")

    rows = []
    for n in N_RANGE:
        counts, depth1 = [], None
        for k in K_PROBES:
            qc = grover_circuit(n, MARKED, k, ancillas=ancillas)
            tq = transpile(qc, **target, optimization_level=OPT_LEVEL,
                           seed_transpiler=SEED)
            counts.append(two_qubit_count(tq))
            if k == 1:
                depth1 = tq.depth()

        slope, intercept, *_ = stats.linregress(K_PROBES, counts)
        rows.append(dict(topology=topology, ancillas=ancillas, n=n, N=2 ** n,
                         **{f"two_qubit_k{k}": c for k, c in zip(K_PROBES, counts)},
                         per_iteration_2q=slope, intercept_2q=intercept,
                         transpiled_depth_k1=depth1))
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

    # ---- power law in n, or exponential in n? -------------------------------
    # A power law in n is polylogarithmic in N = 2**n; an exponential in n
    # would be a power of N. The model with the lower AIC fits better.
    exp_fit = stats.linregress(ns, np.log(gates))
    aic_pow = aic(np.log(gates) - (fit.intercept + fit.slope * np.log(ns)))
    aic_exp = aic(np.log(gates) - (exp_fit.intercept + exp_fit.slope * ns))
    local_b = np.diff(np.log(gates)) / np.diff(np.log(ns))
    print(f"  exponential fit: per-iteration ~ exp({exp_fit.slope:.3f} n)   "
          f"R^2 = {exp_fit.rvalue**2:.4f}")
    print(f"  AIC power law = {aic_pow:.1f}   AIC exponential = {aic_exp:.1f}   "
          f"difference = {aic_exp - aic_pow:+.1f} (positive favours the power law)")
    print("  local exponents between consecutive n: " + "  ".join(
        f"{int(a)}->{int(c)}: {v:.2f}" for a, c, v in zip(ns[:-1], ns[1:], local_b)))

    rows[0]["local_exponent_from_prev_n"] = float("nan")   # no previous n
    for row, v in zip(rows[1:], local_b):
        row["local_exponent_from_prev_n"] = v

    summary = dict(topology=topology, ancillas=ancillas, b=b, ci_lo=lo, ci_hi=hi,
                   r2=fit.rvalue ** 2, prefactor_a=np.exp(fit.intercept),
                   exp_rate_per_qubit=exp_fit.slope, r2_exponential=exp_fit.rvalue ** 2,
                   aic_power=aic_pow, aic_exponential=aic_exp,
                   per_iteration_at_max_n=gates[-1])
    return rows, ns, gates, fit, summary


def main() -> None:
    all_rows, summaries, curves = [], [], {}
    for topology in TOPOLOGIES:
        for ancillas in ANCILLA_ARMS:
            rows, ns, gates, fit, summary = run_arm(topology, ancillas)
            all_rows.extend(rows)
            summaries.append(summary)
            curves[(topology, ancillas)] = (ns, gates, fit)

    print(f"\nSummary: per-iteration two-qubit cost ~ a * n^b   "
          f"(n = {min(N_RANGE)}..{max(N_RANGE)})")
    print(f"{'topology':>11} {'ancillas':>9} {'b':>7} {'95% CI':>18} {'R^2':>8} "
          f"{'2q/iter at max n':>17} {'AIC exp-pow':>12}")
    for s in summaries:
        ci = f"[{s['ci_lo']:.3f}, {s['ci_hi']:.3f}]"
        print(f"{s['topology']:>11} {s['ancillas']:>9} {s['b']:>7.3f} {ci:>18} "
              f"{s['r2']:>8.4f} {s['per_iteration_at_max_n']:>17.1f} "
              f"{s['aic_exponential'] - s['aic_power']:>+12.1f}")

    with open("results/transpile_study.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    with open("results/transpile_exponents.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(summaries)

    plot(curves)
    print("\nWrote results/transpile_study.csv, results/transpile_exponents.csv "
          "and figures/h1_gate_scaling.png")


def plot(curves) -> None:
    """Figure 1: the two heavy-hex arms (no ancilla, 1 ancilla) with their fits.
    The other topologies are summarised in results/transpile_exponents.csv."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

    OKABE_ITO = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9",
                 "green": "#009E73", "blue": "#0072B2", "vermillion": "#D55E00"}

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    # Each arm is told apart by colour AND marker AND line style.
    for ancillas, colour, marker, fit_style, name in (
            (0, "blue", "o", "--", "no ancilla"),
            (1, "vermillion", "s", "-.", "1 ancilla")):
        if ("heavy-hex", ancillas) not in curves:
            continue
        ns, gates, fit = curves[("heavy-hex", ancillas)]
        ax.plot(ns, gates, color=OKABE_ITO[colour], linestyle="none", marker=marker,
                markersize=7, label=f"{name}: measured (transpiled)")
        grid = np.linspace(ns.min(), ns.max(), 200)
        ax.plot(grid, np.exp(fit.intercept) * grid ** fit.slope,
                color=OKABE_ITO[colour], linestyle=fit_style, linewidth=1.8,
                label=fr"{name}: fit $\propto n^{{{fit.slope:.2f}}}$")

    ns, gates, _ = curves[("heavy-hex", 0)]
    grid = np.linspace(ns.min(), ns.max(), 200)
    ax.plot(grid, gates[0] * (grid / ns[0]), color=OKABE_ITO["black"],
            linestyle=":", linewidth=1.5, label="linear reference $n^{1}$")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.xaxis.set_major_locator(FixedLocator(ns))          # ticks at 3, 4, ..., 10
    ax.xaxis.set_major_formatter(FixedFormatter([f"{int(v)}" for v in ns]))
    ax.xaxis.set_minor_locator(NullLocator())             # no "3x10^0" labels
    ax.set_xlabel("qubits $n$")
    ax.set_ylabel("two-qubit gates per Grover iteration")
    ax.grid(alpha=0.25, which="both", linewidth=0.5)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig("figures/h1_gate_scaling.png", dpi=200)


if __name__ == "__main__":
    main()
