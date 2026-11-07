"""
run_cost_model.py
-----------------
Turns the H1 measurement into a PREDICTOR instead of a report.

Three things, in order:

  1. Optimization-level robustness for H1. Refits the scaling exponent b at
     transpiler optimization levels 0-3. Answers the obvious objection that the
     level was picked to flatter the result.
     1b (revision): refits b for every transpiler seed in SEED_SWEEP.

  2. Model validation at every n in KSWEEP_NS, for every ancilla arm.
     Global-depolarizing approximation

         p_hat = F * p_ideal + (1 - F) / N,      F = (1 - eps_2q) ** G

     where G is the transpiled two-qubit gate count from the H1 instrument and
     p_ideal is the closed form. NOTHING IS FITTED: G is measured, eps is the
     noise-model constant, p_ideal is analytic. Checked against the measured
     noisy curves in results/ksweep_n{n}.csv (and ksweep_n{n}_anc1.csv).

  3. H3 projection. Evaluates the same model at n = 3..10 to locate the
     advantage horizon, then inverts it to find the critical eps_2q at which
     each n would first beat classical search. That inversion is the answer to
     the second half of the central question: what error-rate reduction moves
     the boundary out by one qubit.
     Revision: two critical rates are reported, at the best k (any win; in
     practice k = 1, a constant-factor gain) and at the textbook k_th (the
     quadratic regime). G(n, k) is now measured by transpiling every
     k-iteration circuit, not extrapolated from k = 1..3.

Everything in (3) is PROJECTION, not measurement. Label it that way.

Run:   python run_cost_model.py          (after run_transpile_study.py and run_ksweep.py)
Out:   results/cost_model.csv  and  figures/model_validation_n{n}.png (and _anc1)
"""

from __future__ import annotations

import os
import csv

import numpy as np
from scipy import stats, optimize

from qiskit import transpile
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

from grover_core import (
    grover_circuit, ideal_success_probability, k_theoretical, p_threshold,
)

# ---------------------------------------------------------------- config ----
N_RANGE   = range(3, 11)
K_PROBES  = (1, 2, 3)
OPT_LEVEL = 2                 # must match run_transpile_study.py
OPT_SWEEP = (0, 1, 2, 3)
SEED      = 20260815
MARKED    = 0
EPS_2Q    = 8e-3              # must match run_ksweep.py
KSWEEP_NS = (4, 5, 6)         # must match N_LIST in run_ksweep.py
TWO_QUBIT_NAMES = {"cz", "cx", "ecr", "swap", "rzz"}
ANCILLA_ARMS = (0, 1)         # must match run_ksweep.py and run_transpile_study.py
SEED_SWEEP = tuple(range(SEED, SEED + 10))    # 10 transpiler seeds, for 1b
# ----------------------------------------------------------------------------

OKABE_ITO = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9",
             "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
             "vermillion": "#D55E00", "purple": "#CC79A7"}


def two_qubit_count(circuit) -> int:
    return sum(v for k, v in circuit.count_ops().items() if k in TWO_QUBIT_NAMES)


def gate_model(backend, n: int, opt_level: int = OPT_LEVEL, ancillas: int = 0,
               seed: int = SEED):
    """Return (slope, intercept) of transpiled 2q gates vs k, for one n.

    slope is the marginal cost of one Grover iteration; intercept absorbs state
    preparation, measurement, and fixed routing overhead."""
    counts = []
    for k in K_PROBES:
        tq = transpile(grover_circuit(n, MARKED, k, ancillas=ancillas), backend,
                       optimization_level=opt_level, seed_transpiler=seed)
        counts.append(two_qubit_count(tq))
    slope, intercept, *_ = stats.linregress(K_PROBES, counts)
    return slope, intercept


def predict_p(n: int, k: int, G: float, eps: float) -> float:
    """Global-depolarizing approximation. No free parameters.
    G is the transpiled two-qubit gate count of the k-iteration circuit."""
    F = (1.0 - eps) ** G
    return F * ideal_success_probability(n, k) + (1.0 - F) / 2 ** n


_G_CACHE: dict = {}


def measured_G(backend, n: int, k: int, ancillas: int = 0) -> int:
    """Two-qubit gate count of the actual k-iteration circuit, measured by
    transpiling it (not extrapolated from k = 1..3). Cached, so each circuit
    is transpiled only once per run."""
    key = (n, k, ancillas)
    if key not in _G_CACHE:
        tq = transpile(grover_circuit(n, MARKED, k, ancillas=ancillas), backend,
                       optimization_level=OPT_LEVEL, seed_transpiler=SEED)
        _G_CACHE[key] = two_qubit_count(tq)
    return _G_CACHE[key]


def arm_suffix(ancillas: int) -> str:
    """File-name tag of an ancilla arm, as in run_ksweep.py: '' or '_anc1'."""
    return "" if ancillas == 0 else f"_anc{ancillas}"


# ------------------------------------------------------- 1. robustness ------
def robustness(backend, ancillas: int = 0) -> list[dict]:
    print(f"\n[1] H1 robustness to transpiler optimization level   (ancillas = {ancillas})")
    print(f"{'opt':>4} {'b':>7} {'95% CI':>18} {'R^2':>8} {'2q/iter n=10':>13}")
    rows = []
    for opt in OPT_SWEEP:
        slopes = [gate_model(backend, n, opt, ancillas=ancillas)[0] for n in N_RANGE]
        fit = stats.linregress(np.log(list(N_RANGE)), np.log(slopes))
        t_crit = stats.t.ppf(0.975, len(slopes) - 2)
        lo, hi = fit.slope - t_crit * fit.stderr, fit.slope + t_crit * fit.stderr
        rows.append(dict(ancillas=ancillas, opt_level=opt, b=fit.slope, ci_lo=lo,
                         ci_hi=hi, r2=fit.rvalue ** 2, per_iter_n10=slopes[-1]))
        print(f"{opt:>4} {fit.slope:>7.3f} {f'[{lo:.3f}, {hi:.3f}]':>18} "
              f"{fit.rvalue**2:>8.4f} {slopes[-1]:>13.1f}")
    return rows


def seed_robustness(backend, ancillas: int = 0) -> list[dict]:
    """1b. Refit the exponent b for every transpiler seed in SEED_SWEEP."""
    print(f"\n[1b] H1 robustness to the transpiler seed   (ancillas = {ancillas}, "
          f"opt level {OPT_LEVEL})")
    print(f"{'seed':>10} {'b':>7} {'95% CI':>18} {'R^2':>8} {'2q/iter n=10':>13}")
    rows = []
    for seed in SEED_SWEEP:
        slopes = [gate_model(backend, n, OPT_LEVEL, ancillas=ancillas, seed=seed)[0]
                  for n in N_RANGE]
        fit = stats.linregress(np.log(list(N_RANGE)), np.log(slopes))
        t_crit = stats.t.ppf(0.975, len(slopes) - 2)
        lo, hi = fit.slope - t_crit * fit.stderr, fit.slope + t_crit * fit.stderr
        rows.append(dict(ancillas=ancillas, seed=seed, b=fit.slope, ci_lo=lo,
                         ci_hi=hi, r2=fit.rvalue ** 2, per_iter_n10=slopes[-1]))
        ci = f"[{lo:.3f}, {hi:.3f}]"
        print(f"{seed:>10} {fit.slope:>7.3f} {ci:>18} {fit.rvalue**2:>8.4f} "
              f"{slopes[-1]:>13.1f}")
    bs = np.array([r["b"] for r in rows])
    per = np.array([r["per_iter_n10"] for r in rows])
    print(f"    across {len(rows)} seeds: b = {bs.mean():.3f} +/- {bs.std(ddof=1):.3f} "
          f"(min {bs.min():.3f}, max {bs.max():.3f});  2q/iter at n=10 = "
          f"{per.mean():.1f} +/- {per.std(ddof=1):.1f}")
    return rows


# ------------------------------------------------- 2. model validation ------
def validate(backend, n: int, ancillas: int = 0) -> list[dict]:
    path = f"results/ksweep_n{n}{arm_suffix(ancillas)}.csv"
    if not os.path.exists(path):
        print(f"\n[2] SKIPPED - {path} not found. Run run_ksweep.py first.")
        return []

    with open(path) as f:
        meas = list(csv.DictReader(f))

    slope, intercept = gate_model(backend, n)

    print(f"\n[2] Model validation at n = {n}, ancillas = {ancillas}, eps_2q = {EPS_2Q}"
          f"  (zero fitted parameters)")
    print(f"{'k':>3} {'G':>6} {'p_noisy':>9} {'p_pred':>9} {'resid':>9}")

    # Use the two-qubit counts recorded by the k-sweep itself. The H1 regression
    # (slope, intercept) is only needed where no sweep exists -- see project().
    rows, resid = [], []
    for m in meas:
        k = int(m["k"])
        p_meas = float(m["p_noisy"])
        G_k = float(m["two_qubit_gates"])
        F = (1.0 - EPS_2Q) ** G_k
        p_pred = F * ideal_success_probability(n, k) + (1.0 - F) / 2 ** n
        resid.append(p_pred - p_meas)
        rows.append(dict(n=n, ancillas=ancillas, k=k, two_qubit_gates=int(G_k),
                         p_noisy=p_meas, p_noisy_se=float(m.get("p_noisy_se", "nan")),
                         p_pred=p_pred, resid=p_pred - p_meas))
        print(f"{k:>3} {int(G_k):>6} {p_meas:>9.4f} "
              f"{p_pred:>9.4f} {p_pred - p_meas:>+9.4f}")

    resid = np.array(resid)
    ks = [r["k"] for r in rows]
    k_meas = max(rows, key=lambda r: r["p_noisy"])["k"]
    k_pred = max(rows, key=lambda r: r["p_pred"])["k"]
    eq_meas = min(rows, key=lambda r: (r["k"] + 1) / max(r["p_noisy"], 1e-12))["k"]
    eq_pred = min(rows, key=lambda r: (r["k"] + 1) / max(r["p_pred"], 1e-12))["k"]

    print(f"\n    max|resid| = {np.abs(resid).max():.4f}   RMS = {np.sqrt((resid**2).mean()):.4f}")
    print(f"    argmax p:      measured k = {k_meas}   predicted k = {k_pred}")
    print(f"    argmin E[Q]:   measured k = {eq_meas}   predicted k = {eq_pred}")

    # sanity check only: how far is the nominal eps from a fitted one?
    G = np.array([r["two_qubit_gates"] for r in rows], dtype=float)
    p_id = np.array([ideal_success_probability(n, r["k"]) for r in rows])
    p_ms = np.array([r["p_noisy"] for r in rows])

    def sse(e):
        F = (1 - e) ** G
        return np.sum((F * p_id + (1 - F) / 2 ** n - p_ms) ** 2)

    e_fit = optimize.minimize_scalar(sse, bounds=(1e-5, 5e-2), method="bounded").x
    print(f"    best-fit eps_eff = {e_fit:.5f}  ({e_fit / EPS_2Q:.3f} x nominal)")

    plot_validation(ks, p_ms, [r["p_pred"] for r in rows],
                    [float(m["p_threshold"]) for m in meas], n,
                    [r["p_noisy_se"] for r in rows], ancillas)
    return rows


def plot_validation(ks, p_meas, p_pred, p_thresh, n, p_se=None, ancillas=0) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(ks, p_meas, color=OKABE_ITO["vermillion"], linestyle="-.", marker="^",
            markersize=8, linewidth=1.6, label="measured (noisy simulator)")
    if p_se is not None:    # error bars: +-1 standard error (no legend entry)
        ax.errorbar(ks, p_meas, yerr=p_se, fmt="none",
                    ecolor=OKABE_ITO["vermillion"], capsize=3)
    ax.plot(ks, p_pred, color=OKABE_ITO["purple"], linestyle="--", marker="o",
            markersize=6, markerfacecolor="none", markeredgewidth=1.6, linewidth=1.6,
            label=r"predicted  $\hat{p}=F\,p_{\mathrm{ideal}}+(1-F)/N$")
    ax.plot(ks, p_thresh, color=OKABE_ITO["green"], linestyle=":", marker="D",
            markersize=5, linewidth=1.4, label=r"threshold $p^*(N)$")
    ax.axvline(k_theoretical(n), color=OKABE_ITO["black"], linewidth=1, alpha=0.45)
    ax.set_xlabel("Grover iterations $k$")
    ax.set_ylabel("success probability $p$")
    ax.set_ylim(0, max(max(p_meas), max(p_pred)) * 1.2)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out = f"figures/model_validation_n{n}{arm_suffix(ancillas)}"
    fig.savefig(f"{out}.png", dpi=200)
    # greyscale check, same discipline as the other figures
    from PIL import Image
    Image.open(f"{out}.png").convert("L").save(f"{out}_gray.png")


# ---------------------------------------------------- 3. H3 projection ------
def project(backend, ancillas: int = 0) -> list[dict]:
    print(f"\n[3] H3 projection  (MODEL, NOT MEASUREMENT)   ancillas = {ancillas}")
    print("    eps_crit      = error rate at which Grover first wins at ANY k "
          "(k@crit = that k)")
    print("    eps_crit@k_th = error rate at which it wins at the textbook k_th "
          "(quadratic regime)")
    print(f"{'n':>3} {'N':>6} {'k_th':>5} {'p_hat@k_th':>11} {'p*@k_th':>9} "
          f"{'@k_th':>7} {'@best k':>8} {'best k':>7} {'eps_crit':>11} {'k@crit':>7} "
          f"{'factor':>8} {'eps_crit@k_th':>14} {'factor':>8}")

    def G(n, k):
        """Measured two-qubit count of the k-iteration circuit (cached)."""
        return measured_G(backend, n, k, ancillas)

    def margin(n, k, eps):
        """classical E[Q] - quantum E[Q] at one k. > 0 means Grover wins."""
        p = predict_p(n, k, G(n, k), eps)
        return (2 ** n + 1) / 2 - (k + 1) / p if p > 0 else -np.inf

    def best_margin(n, eps):
        """max over k of the margin: does Grover win at ANY k?"""
        kth = k_theoretical(n)
        return max(margin(n, k, eps) for k in range(1, 2 * kth + 3))

    rows, prev, prev_kth = [], None, None
    for n in N_RANGE:
        N, kth = 2 ** n, k_theoretical(n)
        p_kth = predict_p(n, kth, G(n, kth), EPS_2Q)
        ps_kth = p_threshold(n, kth)

        best_k, best_eq = None, np.inf
        for k in range(0, 2 * kth + 3):
            p = predict_p(n, k, G(n, k), EPS_2Q)
            if p > 0 and (k + 1) / p < best_eq:
                best_eq, best_k = (k + 1) / p, k

        # critical rate for ANY win, and the k at which that win happens
        try:
            eps_crit = optimize.brentq(lambda e: best_margin(n, e), 1e-9, 0.5)
            k_at_crit = max(range(1, 2 * kth + 3), key=lambda k: margin(n, k, eps_crit))
        except ValueError:
            eps_crit, k_at_crit = float("nan"), None

        # critical rate for the QUADRATIC regime: the textbook k_th only
        try:
            eps_crit_kth = optimize.brentq(lambda e: margin(n, kth, e), 1e-9, 0.5)
        except ValueError:
            eps_crit_kth = float("nan")

        factor = prev / eps_crit if prev and eps_crit == eps_crit else float("nan")
        factor_kth = (prev_kth / eps_crit_kth
                      if prev_kth and eps_crit_kth == eps_crit_kth else float("nan"))
        prev, prev_kth = eps_crit, eps_crit_kth

        v_kth = "win" if p_kth > ps_kth else "lose"
        v_best = "win" if best_eq < (N + 1) / 2 else "lose"
        rows.append(dict(ancillas=ancillas, n=n, N=N, k_th=kth, p_hat_k_th=p_kth,
                         p_threshold_k_th=ps_kth, verdict_k_th=v_kth, best_k=best_k,
                         expected_queries_best=best_eq, classical_queries=(N + 1) / 2,
                         verdict_best_k=v_best, eps_critical=eps_crit,
                         k_at_eps_critical=k_at_crit, eps_critical_k_th=eps_crit_kth,
                         two_qubit_gates_k_th=G(n, kth)))
        fac = f"{factor:>8.2f}" if factor == factor else f"{'-':>8}"
        fac_kth = f"{factor_kth:>8.2f}" if factor_kth == factor_kth else f"{'-':>8}"
        print(f"{n:>3} {N:>6} {kth:>5} {p_kth:>11.4f} {ps_kth:>9.4f} "
              f"{v_kth:>7} {v_best:>8} {best_k:>7} {eps_crit:>11.3e} {str(k_at_crit):>7} "
              f"{fac} {eps_crit_kth:>14.3e} {fac_kth}")

    wins = [r["n"] for r in rows if r["verdict_best_k"] == "win"]
    if wins:
        print(f"\n    Advantage horizon: last n that wins is {max(wins)}; "
              f"n = {max(wins) + 1} onward loses at every k.")
    else:
        print("\n    No n in range wins at any k.")

    for key, label in (("eps_critical", "any win (best k)"),
                       ("eps_critical_k_th", "quadratic regime (k_th)")):
        facs = [rows[i][key] / rows[i + 1][key] for i in range(len(rows) - 1)]
        print(f"    eps reduction per extra qubit, {label}: "
              f"{min(facs):.2f}x to {max(facs):.2f}x, "
              f"geometric mean {np.exp(np.mean(np.log(facs))):.2f}x")
    return rows


def main() -> None:
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    backend = FakeSherbrooke()
    print(f"backend: {backend.name}   eps_2q = {EPS_2Q}   opt_level = {OPT_LEVEL}   seed = {SEED}")

    rob, seeds, val, proj = [], [], [], []
    for ancillas in ANCILLA_ARMS:
        rob += robustness(backend, ancillas)
    for ancillas in ANCILLA_ARMS:
        seeds += seed_robustness(backend, ancillas)
    for n in KSWEEP_NS:
        for ancillas in ANCILLA_ARMS:
            val += validate(backend, n, ancillas)
    for ancillas in ANCILLA_ARMS:
        proj += project(backend, ancillas)

    with open("results/cost_model.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# section 1: H1 robustness by optimization level"])
        w.writerow(rob[0].keys())
        for r in rob:
            w.writerow(r.values())
        w.writerow([])
        w.writerow(["# section 1b: H1 robustness by transpiler seed"])
        w.writerow(seeds[0].keys())
        for r in seeds:
            w.writerow(r.values())
        if val:
            w.writerow([])
            w.writerow(["# section 2: model validation (each row carries n and ancillas)"])
            w.writerow(val[0].keys())
            for r in val:
                w.writerow(r.values())
        w.writerow([])
        w.writerow(["# section 3: H3 projection (model, not measurement)"])
        w.writerow(proj[0].keys())
        for r in proj:
            w.writerow(r.values())

    print("\nWrote results/cost_model.csv and figures/model_validation_n*.png")


if __name__ == "__main__":
    main()
