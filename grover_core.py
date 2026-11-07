"""
grover_core.py
--------------
Grover's algorithm for arbitrary n, with the iteration count k exposed as a
free parameter (NOT hardcoded to the theoretical optimum).

Everything else in the project imports from this file, so that simulation and
hardware use an identical circuit-construction path.
"""

from __future__ import annotations

import math
from qiskit import QuantumCircuit


# ----------------------------------------------------------------------
# Bit-ordering convention
# ----------------------------------------------------------------------
# Qiskit is little-endian: qubit 0 is the RIGHTMOST character of a bitstring.
# We take `marked` as an integer in [0, 2**n) and convert it once, here, so the
# convention is stated in exactly one place.

#Using 'Type annotations" to declare the funtions
def index_to_bitstring(marked: int, n: int) -> str:
    """Big-endian string, e.g. marked=5, n=3 -> '101'. Matches Qiskit's
    measurement keys, so counts['101'] is the marked item."""
    return format(marked, f"0{n}b")


def _multi_controlled_z(qc: QuantumCircuit, n: int) -> None:
    """Flip the phase of |11...1>. This is the only place a multi-qubit gate
    appears, and it is the thing that gets expensive after transpilation.
    z is the Pauli z gate. cz is controlled-Z. h is Hadamard while mcx is multi-
    controlled X. for n≥3, X sandwiched between 2 H because HXH = Z"""
    if n == 1:
        qc.z(0)
    elif n == 2:
        qc.cz(0, 1)
    else:
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)


def phase_oracle(n: int, marked: int) -> QuantumCircuit:
    """U_f : |x> -> -|x> if x == marked, else |x> -> |x>.

    Non-marked amplitudes are untouched. Implemented by X-conjugating the
    marked pattern into |11...1>, phase-flipping that, and undoing the X's.
    """
    bits = index_to_bitstring(marked, n)
    qc = QuantumCircuit(n, name=f"Oracle({bits})") #The "name" is just a label for display and debugging
    zeros = [i for i in range(n) if bits[n - 1 - i] == "0"]
    if zeros:
        qc.x(zeros)
    _multi_controlled_z(qc, n)
    if zeros:
        qc.x(zeros)
    return qc


def diffuser(n: int) -> QuantumCircuit:
    """The Grover diffusion operator: reflection about the mean amplitude.
    2|s><s| - I, built as H^n (2|0><0| - I) H^n.

    This changes every amplitude, marked and unmarked alike. It is the
    oracle that leaves non-marked states alone.
    """
    qc = QuantumCircuit(n, name="Diffuser")
    qc.h(range(n))
    qc.x(range(n))
    _multi_controlled_z(qc, n)
    qc.x(range(n))
    qc.h(range(n))
    return qc


def grover_circuit(n: int, marked: int, k: int, measure: bool = True,
                   ancillas: int = 0) -> QuantumCircuit:
    """Full Grover circuit with exactly k iterations.

    k is a free parameter on purpose: k=0 (no iterations, uniform random) and
    k > k_theoretical (over-rotation) are both legal and both needed for the
    H2 sweep.

    ancillas adds that many idle qubits after the n search qubits. No gate acts
    on them here; the transpiler may borrow them as clean helper qubits when it
    decomposes the multi-controlled Z. Only the n search qubits are measured.
    """
    if not 0 <= marked < 2 ** n:
        raise ValueError(f"marked must be in [0, {2**n}) for n={n}")
    if k < 0:
        raise ValueError("k must be >= 0")
    if ancillas < 0:
        raise ValueError("ancillas must be >= 0")

    orc = phase_oracle(n, marked)
    dif = diffuser(n)

    qc = QuantumCircuit(n + ancillas, n, name=f"Grover(n={n},k={k})")
    qc.h(range(n))                      # uniform superposition over all N = 2**n
    for _ in range(k):
        qc.compose(orc, range(n), inplace=True)
        qc.compose(dif, range(n), inplace=True)
    if measure:
        qc.measure(range(n), range(n))
    return qc


# ----------------------------------------------------------------------
# Theory, for comparison against measurement
# ----------------------------------------------------------------------

def theta(n: int, num_marked: int = 1) -> float:
    """sin(theta) = sqrt(M/N). Each Grover iteration rotates the state by 2*theta."""
    return math.asin(math.sqrt(num_marked / 2 ** n))


def ideal_success_probability(n: int, k: int, num_marked: int = 1) -> float:
    """p_ideal = sin^2((2k+1) * theta). Non-monotonic in k: it peaks and then falls."""
    return math.sin((2 * k + 1) * theta(n, num_marked)) ** 2


def k_theoretical(n: int, num_marked: int = 1) -> int:
    """k_th = floor((pi/4) * sqrt(N/M))."""
    return math.floor((math.pi / 4) * math.sqrt(2 ** n / num_marked))


def classical_expected_queries(n: int) -> float:
    """(N+1)/2 -- expected queries of linear search, marked item uniform."""
    return (2 ** n + 1) / 2


def quantum_expected_queries(k: int, p: float) -> float:
    """(k+1)/p -- k oracle queries per attempt, +1 classical verification,
    divided by the per-attempt success probability."""
    if p <= 0:
        return float("inf")
    return (k + 1) / p


def p_threshold(n: int, k: int) -> float:
    """p*(N) = 2(k+1)/(N+1). Grover wins iff measured p exceeds this."""
    return 2 * (k + 1) / (2 ** n + 1)


if __name__ == "__main__":
    print(f"{'n':>3} {'N':>6} {'k_th':>5} {'p_ideal':>9} {'p*(N)':>8}  verdict")
    for n in range(2, 11):
        k = k_theoretical(n)
        p = ideal_success_probability(n, k)
        ps = p_threshold(n, k)
        print(f"{n:>3} {2**n:>6} {k:>5} {p:>9.4f} {ps:>8.4f}  "
              f"{'Grover wins' if p > ps else 'classical wins'}")
