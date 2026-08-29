# Where does Grover's quadratic advantage actually end?

An empirical study of the point at which Grover's search algorithm stops beating
classical linear search on real superconducting hardware.

The textbook quadratic speedup rests on two abstractions: that an oracle query
is unit-cost, and that the algorithm runs at its theoretically optimal iteration
count. Neither survives transpilation onto a fixed coupling map or execution
under decoherence. This repository is the instrument for measuring where each
one fails.

**Status:** in progress, week 3 of 6. The circuit implementation and the
k-sweep are running; the transpilation study is written and awaiting execution;
hardware runs and analysis are not yet done.

---

## The claim being tested

On noisy hardware Grover is a repeat-until-success protocol: run, measure a
candidate, verify it with one classical query, repeat on failure. Expected query
cost is therefore `(k + 1) / p`, against classical linear search's `(N + 1) / 2`.
So Grover retains a genuine query advantage if and only if

```
p  >  p*(N)  =  2(k + 1) / (N + 1)
```

The threshold falls as N grows while decoherence rises with circuit depth. The
crossing point is what this measures.

## Hypotheses

Each is stated with the evidence that would refute it. Null results are
reportable findings, not project failures.

| | Hypothesis | Refuted if |
|---|---|---|
| **H1** | Transpiled two-qubit gate count per Grover iteration grows super-linearly in `n` under a fixed coupling map | a power-law fit yields exponent `b ≤ 1.0` with the 95% CI excluding a super-linear value |
| **H2** | For sufficiently large `n`, the empirically optimal iteration count falls strictly below `k_th = ⌊(π/4)√(N/M)⌋` | the two agree within measurement resolution at every tested `n` |
| **H3** | There is some `n` within reach of the device at which measured `p` drops below `p*(N)` | `p` exceeds `p*(N)` at every `n` the hardware can execute |

H1 requires no quantum hardware — transpilation is a compile-time
transformation — so it stands alone as a complete contribution if device access
does not materialise.

---

## Layout

```
grover_core.py            circuit construction + closed-form theory
run_ksweep.py             H2 instrument: sweeps k, ideal vs. noisy
run_transpile_study.py    H1 instrument: gate-count scaling vs. n
results/                  immutable raw CSVs
figures/                  regenerated deterministically from results/
```

`grover_core.py` is the single circuit-construction path. Simulation and
hardware import from it, so the two cannot silently diverge. The iteration count
`k` is a free parameter rather than hardcoded to the optimum, which is what
makes the H2 sweep possible at all.

## Requirements

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Pinned: Python `3.14.7`, Qiskit `2.5.2`, qiskit-aer `0.17.2`,
qiskit-ibm-runtime `0.49.0`, numpy `2.5.2`, scipy `1.18.1`, matplotlib `3.11.1`.

## Running

```bash
python grover_core.py            # theory table: k_th, p_ideal, p*(N) per n
python run_ksweep.py             # -> results/ksweep_n4.csv, figures/ksweep_n4.png
python run_transpile_study.py    # -> results/transpile_study.csv, figures/h1_gate_scaling.png
```

`run_ksweep.py` sweeps `k` from 0 to `2·k_th` at fixed `n`, on an ideal
simulator and on a depolarizing noise model. Circuits are transpiled onto the
heavy-hex device target *first*, so the noisy run carries real routing overhead
rather than assuming it away. It prints `k*_ideal` and `k*_noisy` against
`k_th` — the H2 signal.

`run_transpile_study.py` extracts the marginal two-qubit cost of one Grover
iteration by fitting gate count against `k`, which separates per-iteration cost
from state preparation and measurement, then fits `log(slope)` against `log(n)`
to recover the exponent `b` with a 95% CI. The verdict is stated in advance for
all three outcomes: super-linear, refuted, or inconclusive.

Key parameters are constants at the top of each script.

## Controls

Fixed before implementation began, and logged rather than reconstructed
afterward.

- **Shots:** 4096 per circuit.
- **Placements:** 30 randomized marked-item positions per configuration, fixed
  seed (`20260815`).
- **Transpiler:** optimization level 2, held constant across every comparison.
- **Backend target:** heavy-hex (127-qubit fake provider) for both the routing
  overhead in the noisy sweep and the H1 gate counts.
- **Noise model:** depolarizing, parameterised by 1q and 2q error rates so the
  error rate can be *swept* rather than fixed to one chip. This is what makes
  "what error-rate reduction moves the boundary by one qubit" answerable.
- **Figures:** Okabe–Ito colorblind-safe palette, every series distinguished by
  two channels (color plus marker shape or line style), with a luminance-
  converted copy written alongside each figure to verify greyscale survival.
- **Reproducibility:** raw CSVs immutable; every figure regenerated from them.

`n = 2` is excluded from the H1 sweep deliberately. Grover on `N = 4` is exact
and periodic in `k`, so the transpiler cancels whole iterations away and the
per-iteration slope comes out negative. This is a property of the algorithm, not
a bug in the measurement.

## Scope

Written down in advance, and narrow on purpose: single marked item only; no
error mitigation as a research variable; no cross-device comparison; no
amplitude-amplification variants; no wall-clock head-to-heads. Simulator
wall-clock is recorded but labelled an artifact throughout, and QPU-versus-CPU
timing is never reported as an algorithmic comparison.

## References

- Grover (1996), *A fast quantum mechanical algorithm for database search*
- Bennett, Bernstein, Brassard & Vazirani (1997) — the matching `Ω(√N)` lower
  bound that makes the quadratic separation tight rather than merely best-known
- Boyer, Brassard, Høyer & Tapp (1998) — exact iteration counts
- Babbush, McClean, Newman, Gidney, Boixo & Neven, *PRX Quantum* (2021) — the
  case that error-correction overhead eliminates practical advantage for
  quadratic speedups specifically

---

Sushant Paudel · `passing_hardy_0j@icloud.com` · `Copyright (c) 2026 sushant8099`
