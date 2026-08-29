"""
run_ksweep.py
-------------
Sweeps the Grover iteration count k at fixed n and records measured success
probability, on (a) an ideal simulator and (b) a depolarizing noise model
parameterised by the two-qubit gate error rate.

This is the H2 instrument: it locates the EMPIRICAL optimum k* and compares it
to the theoretical k_th = floor((pi/4) sqrt(N/M)).

Run:   python run_ksweep.py
Out:   results/ksweep_n{N}.csv  and  figures/ksweep_n{N}.png
"""

from __future__ import annotations

import os
import csv
import random

from PIL import Image

from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

from grover_core import (
    grover_circuit, index_to_bitstring, ideal_success_probability,
    k_theoretical, p_threshold, quantum_expected_queries,
    classical_expected_queries,
)

# ---------------------------------------------------------------- config ----
N_QUBITS       = 4          # n. N = 2**n items.
SHOTS          = 4096       # matches the protocol in the write-up
N_PLACEMENTS   = 30         # >= 30 randomized marked-item placements
K_MAX          = None       # None -> sweep to 2 * k_theoretical
OPT_LEVEL      = 2          # FIXED across all comparisons. Log it.
SEED           = 20260815   # fixed seed for randomized placement
EPS_2Q         = 8e-3       # two-qubit depolarizing error rate
EPS_1Q         = 2e-4       # single-qubit depolarizing error rate
BASIS          = ["rz", "sx", "x", "cz"]   # rz is virtual on IBM -> noiseless
# ----------------------------------------------------------------------------


def build_noise_model(eps_1q: float, eps_2q: float) -> NoiseModel:
    """A deliberately simple, parameterised model. The point is not fidelity to
    one specific chip but the ability to SWEEP eps_2q, which is what the
    'error-rate reduction to move the boundary by one qubit' question needs."""
    nm = NoiseModel(basis_gates=BASIS)
    nm.add_all_qubit_quantum_error(depolarizing_error(eps_1q, 1), ["sx", "x"])
    nm.add_all_qubit_quantum_error(depolarizing_error(eps_2q, 2), ["cz"])
    return nm


def measure_p(circuit, backend, marked_bits, shots=SHOTS, seed=None) -> float:
    """Fraction of shots that returned the marked bitstring = empirical p."""
    result = backend.run(circuit, shots=shots, seed_simulator=seed).result()
    counts = result.get_counts()
    return counts.get(marked_bits, 0) / shots


def main() -> None:
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    n = N_QUBITS
    k_th = k_theoretical(n)
    k_max = K_MAX if K_MAX is not None else max(2 * k_th, 4)

    hardware_target = FakeSherbrooke()          # heavy-hex, 127 qubits
    ideal_backend = AerSimulator()
    noisy_backend = AerSimulator(
        noise_model=build_noise_model(EPS_1Q, EPS_2Q), basis_gates=BASIS
    )

    rng = random.Random(SEED)
    placements = [rng.randrange(2 ** n) for _ in range(N_PLACEMENTS)]

    rows = []
    print(f"n={n}  N={2**n}  k_th={k_th}  sweeping k=0..{k_max}")
    print(f"{'k':>3} {'p_ideal_th':>11} {'p_ideal':>9} {'p_noisy':>9} "
          f"{'2q gates':>9} {'E[Q] noisy':>11} {'p*(N)':>7}")

    for k in range(k_max + 1):
        p_ideal_runs, p_noisy_runs = [], []
        twoq = None

        for j, marked in enumerate(placements):
            qc = grover_circuit(n, marked, k)
            bits = index_to_bitstring(marked, n)

            # Ideal: transpile only to the simulator's own basis, no routing.
            qc_ideal = transpile(qc, ideal_backend, optimization_level=OPT_LEVEL,
                                 seed_transpiler=SEED)
            p_ideal_runs.append(measure_p(qc_ideal, ideal_backend, bits, seed=SEED + j))

            # Noisy: transpile to the REAL device target first, so the circuit
            # carries heavy-hex routing overhead, then apply the noise model.
            qc_hw = transpile(qc, hardware_target, optimization_level=OPT_LEVEL,
                              seed_transpiler=SEED)
            qc_noisy = transpile(qc_hw, basis_gates=BASIS, optimization_level=0)
            if twoq is None:
                twoq = qc_noisy.count_ops().get("cz", 0)
            p_noisy_runs.append(measure_p(qc_noisy, noisy_backend, bits, seed=SEED + j))

        p_ideal = sum(p_ideal_runs) / len(p_ideal_runs)
        p_noisy = sum(p_noisy_runs) / len(p_noisy_runs)
        p_th = ideal_success_probability(n, k)
        eq = quantum_expected_queries(k, p_noisy)
        ps = p_threshold(n, k)

        rows.append(dict(n=n, N=2 ** n, k=k, p_theory=p_th, p_ideal=p_ideal,
                         p_noisy=p_noisy, two_qubit_gates=twoq,
                         expected_queries_noisy=eq, p_threshold=ps,
                         classical_queries=classical_expected_queries(n)))
        print(f"{k:>3} {p_th:>11.4f} {p_ideal:>9.4f} {p_noisy:>9.4f} "
              f"{twoq:>9} {eq:>11.2f} {ps:>7.4f}")

    with open(f"results/ksweep_n{n}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    k_star_ideal = max(rows, key=lambda r: r["p_ideal"])["k"]
    k_star_noisy = max(rows, key=lambda r: r["p_noisy"])["k"]
    print(f"\nk_theoretical = {k_th}   k*_ideal = {k_star_ideal}   "
          f"k*_noisy = {k_star_noisy}")
    print("H2 signal: k*_noisy < k_theoretical" if k_star_noisy < k_th
          else "H2 signal: no shift at this n / this error rate")

    plot(rows, n, k_th)
    print(f"\nWrote results/ksweep_n{n}.csv and figures/ksweep_n{n}.png")


def plot(rows, n, k_th) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Okabe-Ito colourblind-safe palette. Every series is distinguished by TWO
    # channels (colour AND marker/linestyle) so the figure survives greyscale
    # printing and any form of colour vision deficiency.
    OKABE_ITO = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9",
                 "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
                 "vermillion": "#D55E00", "purple": "#CC79A7"}

    ks = [r["k"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    ax.plot(ks, [r["p_theory"] for r in rows], color=OKABE_ITO["black"],
            linestyle="-", linewidth=3.5, alpha=0.35,
            label=r"theory $\sin^2((2k+1)\theta)$")
    ax.plot(ks, [r["p_ideal"] for r in rows], color=OKABE_ITO["blue"],
            linestyle="--", marker="s", markersize=5, label="ideal simulator")
    ax.plot(ks, [r["p_noisy"] for r in rows], color=OKABE_ITO["vermillion"],
            linestyle="-.", marker="^", markersize=6, label="noisy simulator")
    ax.plot(ks, [r["p_threshold"] for r in rows], color=OKABE_ITO["green"],
            linestyle=":", marker="D", markersize=5, label=r"threshold $p^*(N)$")

    ax.axvline(k_th, color=OKABE_ITO["black"], linewidth=1, alpha=0.5)
    ax.annotate(r"$k_{\mathrm{th}}$", xy=(k_th, 1.02), ha="center", fontsize=10)

    ax.set_xlabel("Grover iterations $k$")
    ax.set_ylabel("success probability $p$")
    ax.set_title(f"k-sweep at n = {n}  (N = {2**n})")
    ax.set_ylim(0, 1.08)
    ax.grid(alpha=0.25, linestyle="-", linewidth=0.5)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(f"figures/ksweep_n{n}.png", dpi=200)
    Image.open(f"figures/ksweep_n{n}.png").convert("L").save(f"figures/ksweep_n{n}_gray.png")


if __name__ == "__main__":
    main()
