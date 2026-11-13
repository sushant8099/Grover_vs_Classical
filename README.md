# Where does Grover's quadratic advantage actually end?

Code, results and figures for the paper **"Where Grover's Quadratic Advantage
Ends: The Compiled Cost of a Unit-Cost Query"** (Sushant Paudel, 2026).

The textbook quadratic speedup of Grover's search rests on two abstractions:
that an oracle query is unit-cost, and that the algorithm runs at its
theoretically optimal iteration count. Neither survives compilation onto a fixed
coupling map or execution under noise. This repository is the instrument that
measures where each one fails, and predicts the problem size at which Grover
stops beating classical linear search.

**Status:** complete. All results in the paper were produced at commit
`da7dd42`. Everything is simulation: compilation targets are device models, and
noise is a parameterised depolarizing model. No result was run on quantum
hardware.

---

## The claim being tested

Grover with `k` iterations is a repeat-until-success protocol: run, measure a
candidate, verify it with one classical query, repeat on failure. Expected query
cost is therefore `(k + 1) / p`, against classical linear search's `(N + 1) / 2`.
So Grover keeps a genuine query advantage if and only if

```
p  >  p*(N)  =  2(k + 1) / (N + 1)
```

The threshold falls as N grows, while noise accumulates with circuit depth. The
crossing point is what this repository measures.

## Hypotheses

Each was stated, with the evidence that would refute it, before any measurement.

| | Hypothesis | Refuted if |
|---|---|---|
| **H1** | Transpiled two-qubit gate count per Grover iteration grows super-linearly in `n` | a power-law fit gives a 95% CI for the exponent `b` at or below 1.0 |
| **H2** | Noise pushes the optimal iteration count below `k_th = ⌊(π/4)√(N/M)⌋` | the two agree within measurement resolution at every tested `n` |
| **H3** | There is a size `n` beyond which Grover stops beating linear search | it beats linear search at every size simulated or projected |

## Main results

- **H1 holds.** On a 127-qubit heavy-hex model without auxiliary qubits, the
  per-iteration two-qubit cost grows as `n^3.68` (95% CI [3.53, 3.83]) over
  n = 3 to 10. The result is stable across optimization levels 0 to 3 and ten
  transpiler seeds.
- **The growth comes mostly from synthesis, not routing.** One idle qubit lets
  the compiler use a linear-size multi-controlled-gate construction. That cuts
  the cost at n = 10 by 7.2× and the exponent to 1.99. Routing contributes a
  factor of only 1.6 to 2.4.
- **H2 holds** at n = 4, 5 and 6, with and without the auxiliary qubit.
- **H3 holds.** A global-depolarizing model with no fitted parameters matches
  the measured sweeps to within 0.04 in success probability. At a two-qubit
  depolarizing parameter of 8 × 10⁻³, it projects that the quadratic regime ends
  after n = 3 without the auxiliary qubit and after n = 4 with it.

---

## Layout

```
grover_core.py            circuit construction + closed-form theory
run_transpile_study.py    H1: gate-count scaling vs. n (3 coupling maps x 2 ancilla arms)
run_ksweep.py             H2: sweeps k on ideal and noisy simulators (n = 4, 5, 6)
run_cost_model.py         robustness checks, model validation, advantage-horizon projection
requirements.txt          full pinned dependency lock
results/                  immutable raw CSVs
figures/                  regenerated deterministically from results/
LICENSE                   MIT
```

`grover_core.py` is the single circuit-construction path, so every experiment
builds circuits the same way. The iteration count `k` is a free parameter rather
than hardcoded to the optimum, which is what makes the H2 sweep possible.

## Requirements

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Key pins: Python `3.14.7`, Qiskit `2.5.2`, qiskit-aer `0.17.2`,
qiskit-ibm-runtime `0.49.0`, numpy `2.5.2`, scipy `1.18.1`, matplotlib `3.11.1`.
`requirements.txt` pins every dependency.

## Running

Run the scripts in this order. `run_cost_model.py` reads the sweep CSVs.

```bash
python grover_core.py            # theory table: k_th, p_ideal, p*(N) per n
python run_transpile_study.py    # -> results/transpile_study.csv, results/transpile_exponents.csv,
                                 #    figures/h1_gate_scaling.png
python run_ksweep.py             # -> results/ksweep_n{4,5,6}.csv and ksweep_n{4,5,6}_anc1.csv,
                                 #    plus matching figures
python run_cost_model.py         # -> results/cost_model.csv, figures/model_validation_n{4,5,6}[_anc1].png
```

**Single-compile sensitivity check.** Set `NOISY_COMPILE = "direct"` at the top
of `run_ksweep.py` and run it again. This writes the `*_direct` files.

Key parameters are constants at the top of each script.

### What each script does

- **`run_transpile_study.py`** fits gate count against `k` (k = 1, 2, 3). That
  separates the cost of one iteration from state preparation, measurement and
  fixed routing overhead. It then fits `log(slope)` against `log(n)` to recover
  the exponent `b` with a 95% CI, and compares a power law with an exponential
  fit by AIC. It runs on three targets, each with 0 or 1 idle auxiliary qubit:
  heavy-hex (`FakeSherbrooke`, 127 qubits), a square lattice (`FakeNighthawk`,
  120 qubits), and all-to-all (no routing, so synthesis alone).
- **`run_ksweep.py`** sweeps `k` from 0 to `2·k_th` against the closed form, an
  ideal simulator and a depolarizing noise model. Noisy circuits are transpiled
  onto the heavy-hex target *first*, so they carry real routing overhead. Every
  possible marked item is used once (N ≤ 64), and standard errors across
  placements are recorded.
- **`run_cost_model.py`** does three things:
  - repeats the exponent fit at optimization levels 0 to 3 and ten transpiler
    seeds
  - validates the parameter-free model `p̂ = F·p_ideal + (1 − F)/N`, with
    `F = (1 − ε₂q)^G`, against every sweep
  - projects the advantage horizon for n = 3 to 10, including the critical error
    rates

## Controls

- **Shots:** 4096 per circuit.
- **Placements:** every marked item once when N ≤ 64, which covers all swept
  sizes.
- **Seeds:** transpiler and simulator seed `20260815`. The simulator seed is
  offset by the placement index.
- **Transpiler:** optimization level 2 for every comparison, except the
  robustness check that varies it.
- **Noise model:** Qiskit Aer depolarizing errors, with parameter `ε₂q = 8e-3`
  on two-qubit gates and `2e-4` on `sx` and `x`. `rz` is virtual, so it is
  noiseless. For a two-qubit channel, average gate infidelity is `3ε/4`.
- **Figures:** Okabe–Ito colorblind-safe palette. Every series is distinguished
  by two channels (color plus marker shape or line style). Each figure has a
  greyscale copy (`*_gray.png`) to verify it still reads without color.
- **Reproducibility:** raw CSVs are immutable, and every figure is regenerated
  from them.

`n = 2` is excluded from the H1 sweep on purpose. Grover on `N = 4` is exact and
periodic in `k`, so the transpiler cancels whole iterations and the
per-iteration slope comes out negative. That is a property of this special case,
not a bug in the measurement.

## Scope

Narrow on purpose, and written down in advance:
- single marked item only
- no error mitigation as a research variable
- no hardware runs or cross-device comparison (coupling maps are compared at
  compile time only)
- no amplitude-amplification variants
- no wall-clock comparisons

## How to cite

> S. Paudel, *Where Grover's Quadratic Advantage Ends: The Compiled Cost of a Unit-Cost Query, and the Spare Qubit That Halves It*, arXiv:XXXX.XXXXX (2026).
> Code and data: https://doi.org/10.5281/zenodo.24202991

## Key references

- Grover (1996), *A fast quantum mechanical algorithm for database search*
- Bennett, Bernstein, Brassard & Vazirani (1997): the matching `Ω(√N)` lower bound
- Boyer, Brassard, Høyer & Tapp (1998): exact iteration counts
- Wang & Krstic, *Phys. Rev. A* 102, 042609 (2020): noise thresholds for Grover in Qiskit
- Babbush et al., *PRX Quantum* 2, 010103 (2021): error-correction overhead and quadratic speedups
- Khattar & Gidney, *Quantum* 9, 1752 (2025): the one-clean-ancilla construction Qiskit selects

The full reference list is in the paper.

---

Sushant Paudel · passing_hardy_0j@icloud.com · ORCID [0009-0006-8766-9883](https://orcid.org/0009-0006-8766-9883) · MIT License (see `LICENSE`)
