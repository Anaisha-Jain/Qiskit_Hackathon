import numpy as np
from scipy.linalg import expm
import matplotlib
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp, Operator  #Statevector: Statevector(qc) computes the exact quantum state on your laptop, with no sampling noise
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter, SuzukiTrotter



def day1_qiskit_basics():
    qc = QuantumCircuit(2)
    qc.h(0)         
    qc.cx(0, 1)      
    print("Bell state probabilities:", Statevector(qc).probabilities_dict())

    H = SparsePauliOp.from_list([("ZZ", 1.0), ("XI", 0.5), ("IX", 0.5)])  # Hamiltonian H is the “energy rule” of a system, and it determines how the state changes in time
    print("\nHamiltonian as Pauli terms:\n", H)
    print("As a matrix:\n", np.round(H.to_matrix().real, 2))

    init = Statevector.from_label("00")
    t = 3.0
    exact = expm(-1j * H.to_matrix() * t) @ init.data      

    print("\nTrotter fidelity vs exact (1.0 = perfect):")
    for reps in [1, 5, 20]:
        gate = PauliEvolutionGate(H, time=t, synthesis=LieTrotter(reps=reps))
        qc = QuantumCircuit(2)
        qc.append(gate, [0, 1])
        approx = init.evolve(qc.decompose(reps=2))
        fidelity = abs(approx.data.conj() @ exact) ** 2
        print(f"  reps={reps:>2}  fidelity={fidelity:.4f}")


def build_hamiltonian(n_qubits=4, L=2.0, x0=1.0, barrier=4.0, asym=0.0, mass=1.0):
    N = 2 ** n_qubits
    x = np.linspace(-L, L, N)
    dx = x[1] - x[0]
    off = np.ones(N - 1)
    # Kinetic energy: finite-difference second derivative (tridiagonal)
    T = (np.diag(2 * np.ones(N)) - np.diag(off, 1) - np.diag(off, -1)) / (2 * mass * dx**2)
    # Potential: wells at x = +/- x0, barrier at 0, plus linear tilt
    V = barrier * ((x / x0) ** 2 - 1) ** 2 + asym * (x / x0)
    return x, T + np.diag(V), V


def left_well_state(x, x0=1.0, width=0.35):
    """Normalized Gaussian centered on the left well."""
    psi = np.exp(-((x + x0) ** 2) / (2 * width**2)).astype(complex)
    return psi / np.linalg.norm(psi)


def p_right(psi, x):
    """Probability of finding the proton in the right well (x > 0)."""
    return np.sum(np.abs(psi[x > 0]) ** 2)


def evolve_probs(H, psi0, x, times):
    """P(right well) over time, using the eigendecomposition of H."""
    E, U = np.linalg.eigh(H)
    c = U.T @ psi0                                   # state in energy basis
    return [p_right(U @ (np.exp(-1j * E * t) * c), x) for t in times]


def day2_double_well():
    print("\n=== DAY 2: classical double-well model ===")
    x, H, V = build_hamiltonian(asym=0.0)
    E, U = np.linalg.eigh(H)
    print("Lowest levels:", np.round(E[:4], 3))
    print("Tunneling splitting E1-E0:", round(E[1] - E[0], 4))


    for b in [2.0, 4.0, 8.0]:
        Eb = np.linalg.eigvalsh(build_hamiltonian(barrier=b)[1])
        print(f"  barrier={b}: splitting={Eb[1]-Eb[0]:.3f}")

    times = np.linspace(0, 20, 400)
    psi0 = left_well_state(x)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))


    ax[0].plot(x, V, "k-o", ms=3, label="potential")
    for i in range(2):
        sign = np.sign(U[np.argmax(abs(U[:, i])), i])
        ax[0].plot(x, E[i] + 6 * U[:, i] * sign, label=f"state {i}")
    ax[0].set_ylim(-1, 12)
    ax[0].set_title("Potential + lowest 2 states")
    ax[0].legend()

    for asym, label in [(0.0, "symmetric"), (0.8, "asymmetric")]:
        xa, Ha, _ = build_hamiltonian(asym=asym)
        probs = evolve_probs(Ha, psi0, xa, times)
        ax[1].plot(times, probs, label=label)
        print(f"{label}: max P(right) = {max(probs):.3f}")

        direct = p_right(expm(-1j * Ha * 7.0) @ psi0, xa)
        fast = evolve_probs(Ha, psi0, xa, [7.0])[0]
        assert abs(direct - fast) < 1e-9, "eigen-evolution disagrees with expm!"

    ax[1].set_xlabel("time")
    ax[1].set_ylabel("P(proton in right well)")
    ax[1].set_title("Tunneling dynamics")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig("day2_double_well.png", dpi=120)
    print("Saved plot -> day2_double_well.png")
    plt.show()   # pop-up window (close it to end the script)


# =====================================================================
# DAY 3: THE QUANTUM VERSION (Trotterized circuits vs exact)
# =====================================================================
def trotter_step(op, n_qubits, dt, formula):
    """Build the circuit for ONE Trotter step of size dt.
    Returns (circuit, matrix) where matrix is the 2^n x 2^n unitary that circuit implements."""
    qc = QuantumCircuit(n_qubits)
    qc.append(PauliEvolutionGate(op, time=dt, synthesis=formula), range(n_qubits))
    # IMPORTANT: .decompose() turns the gate into real CNOT/rotation gates.
    # Without it, Operator()/Statevector would use the EXACT matrix and hide the Trotter error.
    qc = qc.decompose(reps=2)
    return qc, Operator(qc).data


def trotter_probs(op, n_qubits, psi0, x, t_max, n_steps, formula):
    """P(right well) over time when the evolution is done by n_steps applications of a Trotter circuit."""
    dt = t_max / n_steps
    qc, U = trotter_step(op, n_qubits, dt, formula)    # build the circuit ONCE
    psi = psi0.copy()
    probs = [p_right(psi, x)]
    for _ in range(n_steps):                           # applying the circuit n_steps times
        psi = U @ psi
        probs.append(p_right(psi, x))
    return np.linspace(0, t_max, n_steps + 1), np.array(probs), qc


def day3_trotter():
    print("\n=== DAY 3: quantum (Trotter) simulation of the double well ===")
    n_qubits, t_max = 4, 20.0
    x, H, V = build_hamiltonian(n_qubits=n_qubits, asym=0.0)
    psi0 = left_well_state(x)

    # --- Step 1: classical matrix -> qubit operator (the "map" step) ---
    op = SparsePauliOp.from_operator(H)
    print(f"Hamiltonian: {2**n_qubits}x{2**n_qubits} matrix -> {len(op)} Pauli terms on {n_qubits} qubits")
    print("  conversion error vs original matrix:", f"{np.abs(op.to_matrix() - H).max():.1e}")
    top = sorted(zip(op.paulis.to_labels(), op.coeffs), key=lambda p: -abs(p[1]))[:5]
    print("  biggest terms:", [(lab, round(c.real, 2)) for lab, c in top])

    # --- Step 2: look at one Trotter step as a real circuit ---
    qc1, U1 = trotter_step(op, n_qubits, 0.05, LieTrotter(reps=1))
    sv = Statevector(psi0).evolve(qc1)                  # run the actual circuit on the statevector
    assert np.allclose(sv.data, U1 @ psi0), "circuit and its matrix disagree"
    print(f"One Lie-Trotter step: depth={qc1.depth()}, CNOTs={qc1.count_ops().get('cx', 0)}, "
          f"total gates={sum(qc1.count_ops().values())}")

    # --- Step 3: Trotter dynamics vs exact (Day 2) ---
    formulas = {
        "Lie (1st order)":    lambda: LieTrotter(reps=1),
        "Suzuki order 2":     lambda: SuzukiTrotter(order=2, reps=1),
        "Suzuki order 4":     lambda: SuzukiTrotter(order=4, reps=1),
    }
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

    times_ex = np.linspace(0, t_max, 800)
    ax[0].plot(times_ex, evolve_probs(H, psi0, x, times_ex), "k-", lw=2.5, label="exact (Day 2)")
    for name, n_steps in [("Lie (1st order)", 100), ("Lie (1st order)", 1600), ("Suzuki order 4", 400)]:
        t, p, _ = trotter_probs(op, n_qubits, psi0, x, t_max, n_steps, formulas[name]())
        ax[0].plot(t, p, "--", label=f"{name}, {n_steps} steps")
    ax[0].set_xlabel("time"); ax[0].set_ylabel("P(proton in right well)")
    ax[0].set_title("Trotter circuits vs exact"); ax[0].legend(fontsize=8)

    # --- Step 4: error vs number of steps, and what it costs in circuit depth ---
    step_list = [100, 200, 400, 800, 1600, 3200]
    print("\nMax error in P(right) vs number of Trotter steps:")
    print("  steps      " + "".join(f"{n:>9}" for n in step_list))
    for name, mk in formulas.items():
        errs, depth = [], None
        for n_steps in step_list:
            t, p, qc = trotter_probs(op, n_qubits, psi0, x, t_max, n_steps, mk())
            errs.append(np.max(np.abs(p - evolve_probs(H, psi0, x, t))))
            depth = qc.depth()
        ax[1].loglog(step_list, errs, "o-", label=f"{name} (depth/step={depth})")
        print(f"  {name:<16}" + "".join(f"{e:>9.4f}" for e in errs))
        ok = [n for n, e in zip(step_list, errs) if e < 0.05]
        if ok:
            print(f"    -> error < 0.05 needs {ok[0]} steps ~ total circuit depth {ok[0] * depth:,}")
    ax[1].set_xlabel("Trotter steps"); ax[1].set_ylabel("max |error| in P(right)")
    ax[1].set_title("Error vs steps"); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig("day3_trotter.png", dpi=120)
    print("Saved plot -> day3_trotter.png")

    # --- Step 5: scaling with qubit count (for your "why quantum" slide) ---
    print("\nScaling with grid size (one Lie-Trotter step):")
    print("  qubits  grid pts  Pauli terms  CNOTs/step  depth/step")
    for nq in [3, 4, 5, 6]:
        _, Hn, _ = build_hamiltonian(n_qubits=nq)
        opn = SparsePauliOp.from_operator(Hn)
        qcn, _ = trotter_step(opn, nq, 0.05, LieTrotter(reps=1))
        print(f"  {nq:>6}  {2**nq:>8}  {len(opn):>11}  {qcn.count_ops().get('cx', 0):>10}  {qcn.depth():>10}")
    plt.show()


if __name__ == "__main__":
    day1_qiskit_basics()
    day2_double_well()
    day3_trotter()