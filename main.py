"""
DNA proton-tunneling simulator  (Qiskit hackathon project)
All project code lives in this one file. Run:  python main.py

SETUP (one time, see requirements.txt):
    python -m venv .venv
    .venv\\Scripts\\activate          (Windows)   |   source .venv/bin/activate  (macOS/Linux)
    pip install -r requirements.txt

Day 1: Qiskit basics (circuits, SparsePauliOp, Trotter vs exact)
Day 2: classical double-well model (the ground truth for later days)
Day 3: quantum version -- SparsePauliOp + Trotterized circuits vs the exact curve
Day 4: real physics -- literature A-T / G-C proton-transfer potentials in atomic units
Day 5: open quantum system -- thermal bath, tautomer population & rate vs temperature

Run everything:   python main.py        Run chosen days:   python main.py 4 5
"""
import warnings
import sys
import numpy as np
from scipy.linalg import expm, solve
from scipy.optimize import brentq
import matplotlib
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp, Operator  #Statevector: Statevector(qc) computes the exact quantum state on your laptop, with no sampling noise
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter, SuzukiTrotter

warnings.filterwarnings("ignore")


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


def p_right(psi, x, x_split=0.0):
    """Probability of finding the proton to the right of x_split (the tautomer well)."""
    return np.sum(np.abs(psi[x > x_split]) ** 2)


def evolve_probs(H, psi0, x, times, x_split=0.0):
    """P(right well) over time, using the eigendecomposition of H."""
    E, U = np.linalg.eigh(H)
    c = U.T @ psi0                                   # state in energy basis
    return [p_right(U @ (np.exp(-1j * E * t) * c), x, x_split) for t in times]


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


def trotter_probs(op, n_qubits, psi0, x, t_max, n_steps, formula, x_split=0.0):
    """P(right well) over time when the evolution is done by n_steps applications of a Trotter circuit."""
    dt = t_max / n_steps
    qc, U = trotter_step(op, n_qubits, dt, formula)    # build the circuit ONCE
    psi = psi0.copy()
    probs = [p_right(psi, x, x_split)]
    for _ in range(n_steps):                           # applying the circuit n_steps times
        psi = U @ psi
        probs.append(p_right(psi, x, x_split))
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

# =====================================================================
# DAY 4: REAL PHYSICS -- literature proton-transfer potentials (atomic units)
# =====================================================================
# Atomic units: hbar = m_e = e = 1. Energy in Hartree, length in bohr, time in 24.19 attoseconds.
HARTREE_EV = 27.211386          # 1 Hartree in eV
PROTON_MASS = 1836.152673       # proton mass in atomic units
KB_AU = 3.166811563e-6          # Boltzmann constant, Hartree per kelvin
KB_EV = 8.617333262e-5          # Boltzmann constant, eV per kelvin
H_EV_S = 4.135667696e-15        # Planck constant, eV*s
AU_TIME_FS = 0.02418884254      # 1 atomic unit of time in femtoseconds
BOHR_ANGSTROM = 0.529177210903


def gc_morse_pes(q):
    """G-C proton-transfer potential (Hartree) as a back-to-back Morse potential.
    Parameters from Slocombe, Sacchi & Al-Khalili, Commun. Phys. 5, 109 (2022).
    Canonical well near q=-2.7 bohr, tautomer well near q=+2.1 bohr."""
    v1, v2, a1, a2, r1, r2 = 0.1617, 0.082, 0.305, 0.755, -2.7, 2.1
    return (v1 * (np.exp(-2*a1*(q - r1)) - 2*np.exp(-a1*(q - r1)))
            + v2 * (np.exp(-2*a2*(r2 - q)) - 2*np.exp(-a2*(r2 - q))))


def barrier_fit_pes(Ef_eV, Er_eV, x_left=-2.7, x_right=2.1):
    """Smooth asymmetric double well (a quartic) with minima at x_left (canonical) and x_right
    (tautomer), whose forward barrier is Ef and reverse barrier is Er (both in eV).
    Used for A-T and G-C with the DFT barriers of Slocombe et al., PCCP 23, 4141 (2021), Table 1.
    ASSUMPTION: the well positions (-2.7, +2.1 bohr) are borrowed from the G-C Morse model."""
    Ef, Er = Ef_eV / HARTREE_EV, Er_eV / HARTREE_EV
    dE = Ef - Er                                    # tautomer minus canonical energy

    def primitive(xb):
        # V'(x) = 4a (x - xL)(x - xb)(x - xR): integrate the cubic with np.poly1d
        return np.poly1d(np.poly([x_left, xb, x_right])).integ()

    def height(x, xb):                              # V(x)/(4a) relative to the left minimum
        P = primitive(xb)
        return P(x) - P(x_left)

    # xb must lie beyond the point where both wells have equal energy, else dE would be negative
    x_equal = brentq(lambda xb: height(x_right, xb), x_left + 1e-3, x_right - 1e-6)
    xb = brentq(lambda xb: height(xb, xb) / height(x_right, xb) - Ef / dE, x_equal + 1e-6, x_right - 1e-6)
    a4 = Ef / height(xb, xb)                        # this is 4a
    P = primitive(xb)
    return lambda x: a4 * (P(x) - P(x_left))


def stationary_points(pot, lo=-3.6, hi=3.0):
    """Locate (canonical minimum, barrier top, tautomer minimum) of a double-well potential."""
    q = np.linspace(lo, hi, 60001)
    dV = np.gradient(pot(q), q)
    idx = np.where(np.diff(np.sign(dV)) != 0)[0]
    assert len(idx) == 3, f"expected 3 stationary points, found {len(idx)}"
    return q[idx[0]], q[idx[1]], q[idx[2]]


# name -> (potential function, literature (Ef, Er, dE) in eV)
PRESETS = {
    "G-C (Morse 2022)": (gc_morse_pes, (0.705, 0.270, 0.435)),
    "G-C":              (barrier_fit_pes(0.693, 0.266), (0.693, 0.266, 0.427)),
    "A-T":              (barrier_fit_pes(0.581, 0.011), (0.581, 0.011, 0.570)),
}


def build_pes_hamiltonian(name, n_qubits=6, x_min=-4.5, x_max=3.9, v_cap_eV=3.0):
    """Grid Hamiltonian H = T + V for the proton on the chosen potential.
    v_cap_eV clips the steep outer walls (the proton never gets there) so the Hamiltonian's
    largest energy -- and therefore the Trotter cost -- stays small."""
    pot, _ = PRESETS[name]
    N = 2 ** n_qubits
    x = np.linspace(x_min, x_max, N)
    dx = x[1] - x[0]
    off = np.ones(N - 1)
    T = (np.diag(2*np.ones(N)) - np.diag(off, 1) - np.diag(off, -1)) / (2 * PROTON_MASS * dx**2)
    V = pot(x)
    V = np.minimum(V, V.min() + v_cap_eV / HARTREE_EV)
    xl, xb, xr = stationary_points(pot)
    return x, T + np.diag(V), V, dict(x_canon=xl, x_barrier=xb, x_taut=xr)


def excited_wavepacket(x, H, x_canon, x_barrier, e_exc_eV, width=0.3):
    """Gaussian wavepacket in the canonical well, placed so it carries e_exc_eV of energy above
    the ground state (as if a thermal fluctuation kicked the H-bond vibration)."""
    E0 = np.linalg.eigvalsh(H)[0]

    def make(q0):
        psi = np.exp(-(x - q0) ** 2 / (4 * width**2)).astype(complex)
        return psi / np.linalg.norm(psi)

    excess = lambda q0: (np.vdot(make(q0), H @ make(q0)).real - E0) * HARTREE_EV - e_exc_eV
    q0 = brentq(excess, x_canon, x_barrier)
    return make(q0), q0


def day4_real_potentials(e_exc_eV=0.55):
    print("\n=== DAY 4: real A-T / G-C proton-transfer potentials ===")
    n_qubits = 6
    results = {}

    # --- 1. does each potential reproduce the published barriers? ---
    print("Potential check (eV): model vs literature")
    for name, (pot, (Ef, Er, dE)) in PRESETS.items():
        xl, xb, xr = stationary_points(pot)
        m_f, m_r = (pot(xb) - pot(xl)) * HARTREE_EV, (pot(xb) - pot(xr)) * HARTREE_EV
        m_d = (pot(xr) - pot(xl)) * HARTREE_EV
        print(f"  {name:<18} forward {m_f:.3f}/{Ef:.3f}   reverse {m_r:.3f}/{Er:.3f}   asymmetry {m_d:.3f}/{dE:.3f}")

    # --- 2. is a 6-qubit (64-point) grid fine enough? ---
    print("\nGrid convergence, G-C (Morse): first excitation energies (eV above ground state)")
    for nq in (5, 6, 7, 8):
        _, Hn, _, _ = build_pes_hamiltonian("G-C (Morse 2022)", n_qubits=nq)
        En = np.linalg.eigvalsh(Hn)
        print(f"  {nq} qubits ({2**nq:>3} pts): {np.round((En[1:4] - En[0]) * HARTREE_EV, 4)}")

    # --- 3. spectrum and where each energy level lives ---
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for name, color in [("G-C", "tab:blue"), ("A-T", "tab:red")]:
        x, H, V, info = build_pes_hamiltonian(name, n_qubits)
        E, U = np.linalg.eigh(H)
        w_right = (U[x > info["x_barrier"], :] ** 2).sum(axis=0)
        results[name] = dict(x=x, H=H, V=V, info=info, E=E, U=U)
        print(f"\n{name}: lowest levels (eV above ground state) and fraction of each state in the tautomer well")
        for k in range(10):
            print(f"  level {k}: {(E[k]-E[0])*HARTREE_EV:6.3f} eV   tautomer-well weight {w_right[k]:.4f}")
        xa = x * BOHR_ANGSTROM
        ax[0].plot(xa, (V - V.min()) * HARTREE_EV, color=color, label=f"{name} potential")
        for k in range(14):
            ax[0].hlines((E[k]-V.min())*HARTREE_EV, xa[0], xa[-1], colors=color, lw=0.4, alpha=0.5)
    ax[0].set_ylim(0, 1.0); ax[0].set_xlabel("proton coordinate (Å)"); ax[0].set_ylabel("energy (eV)")
    ax[0].set_title("Literature potentials + energy levels"); ax[0].legend()

    # --- 4. the quantum mapping of the REAL Hamiltonian ---
    x, H, V, info = results["G-C"]["x"], results["G-C"]["H"], results["G-C"]["V"], results["G-C"]["info"]
    op = SparsePauliOp.from_operator(H)
    qc1, _ = trotter_step(op, n_qubits, 1.0, LieTrotter(reps=1))
    print(f"\nG-C Hamiltonian on {n_qubits} qubits: {len(op)} Pauli terms; "
          f"one Lie-Trotter step = {qc1.count_ops().get('cx', 0)} CNOTs, depth {qc1.depth()}")

    # --- 5. coherent tunneling of an excited proton: exact vs Trotter circuits ---
    t_max, n_steps = 6000.0, 1500                    # 6000 a.u. = 145 fs
    print(f"\nCoherent dynamics: proton given {e_exc_eV} eV of vibrational energy, {t_max*AU_TIME_FS:.0f} fs")
    for name, color in [("G-C", "tab:blue"), ("A-T", "tab:red")]:
        r = results[name]
        psi0, q0 = excited_wavepacket(r["x"], r["H"], r["info"]["x_canon"], r["info"]["x_barrier"], e_exc_eV)
        xs = r["info"]["x_barrier"]
        t_ex = np.linspace(0, t_max, 600)
        p_ex = evolve_probs(r["H"], psi0, r["x"], t_ex, xs)
        opn = SparsePauliOp.from_operator(r["H"])
        t_tr, p_tr, qc = trotter_probs(opn, n_qubits, psi0, r["x"], t_max, n_steps, SuzukiTrotter(order=4, reps=1), xs)
        err = np.max(np.abs(p_tr - evolve_probs(r["H"], psi0, r["x"], t_tr, xs)))
        print(f"  {name}: start at q={q0:.2f} bohr, max P(tautomer side)={max(p_ex):.3f}, "
              f"Trotter(order 4, {n_steps} steps) max error={err:.1e}, depth/step={qc.depth()}")
        ax[1].plot(t_ex * AU_TIME_FS, p_ex, color=color, label=f"{name} exact")
        ax[1].plot(t_tr[::25] * AU_TIME_FS, p_tr[::25], "o", color=color, ms=3, label=f"{name} Trotter circuit")
        results[name]["p_max"] = max(p_ex)
    ax[1].set_xlabel("time (fs)"); ax[1].set_ylabel("P(proton on tautomer side)")
    ax[1].set_title(f"Coherent tunneling, {e_exc_eV} eV excitation"); ax[1].legend(fontsize=8)

    # --- 6. why this matters: where the levels sit relative to the barrier ---
    for name, color in [("G-C", "tab:blue"), ("A-T", "tab:red")]:
        r = results[name]
        w = (r["U"][r["x"] > r["info"]["x_barrier"], :] ** 2).sum(axis=0)
        ax[2].semilogy(np.arange(30), np.maximum(w[:30], 1e-12), "o-", color=color, label=name)
    ax[2].set_xlabel("energy level index"); ax[2].set_ylabel("weight on tautomer side")
    ax[2].set_title("Which levels reach the tautomer well"); ax[2].legend()
    plt.tight_layout(); plt.savefig("day4_real_potentials.png", dpi=120)
    print("Saved plot -> day4_real_potentials.png")
    plt.show()
    return results


# =====================================================================
# DAY 5: OPEN QUANTUM SYSTEM -- the proton in a thermal bath
# =====================================================================
# The coherent model (Days 1-4) oscillates forever and ignores temperature. A real H-bond proton is
# constantly jostled by its surroundings. We model that with a thermal bath coupled to the proton's
# position x (Ohmic spectral density, weak-coupling / secular master equation in the energy basis):
#     rate(i -> j) = 2*pi*eta*w_ij*|<i|x|j>|^2 * (n(w_ij) + 1)     downhill (emits energy)
#     rate(j -> i) = 2*pi*eta*w_ij*|<i|x|j>|^2 *  n(w_ij)          uphill   (absorbs energy)
# with n(w) = 1/(exp(w/kT) - 1). The bath strength eta is set by the vibrational relaxation time.
ETA_DEFAULT = 0.01            # illustrative; gives ~1.4 ps relaxation of the first excited level at 300 K
N_BATH_STATES = 40            # number of lowest energy levels kept in the master equation

# Published lifetimes at 298.15 K for comparison (Slocombe et al., PCCP 23, 4141, Table 2), seconds:
# (forward/canonical lifetime, reverse/tautomer lifetime) for WKB and wave-packet-scattering tunnelling.
LIT_LIFETIMES = {
    "G-C": {"WKB": (1.165e-2, 5.486e-10), "scattering": (6.739e-4, 3.175e-11)},
    "A-T": {"WKB": (9.768e-4, 1.752e-13), "scattering": (6.607e-5, 1.185e-14)},
}


def eigen_data(name, n_qubits=6):
    """Everything the open-system model needs: levels, states, tautomer weights, <i|x|j>."""
    x, H, V, info = build_pes_hamiltonian(name, n_qubits)
    E, U = np.linalg.eigh(H)
    w_taut = (U[x > info["x_barrier"], :] ** 2).sum(axis=0)
    X = U.T @ np.diag(x) @ U
    return dict(name=name, x=x, E=E, U=U, w=w_taut, X=X, info=info)


def gibbs_tautomer_population(d, T):
    """Equilibrium probability that the proton sits on the tautomer side at temperature T (kelvin)."""
    p = np.exp(-(d["E"] - d["E"][0]) / (KB_AU * T))
    return (p / p.sum() * d["w"]).sum()


def bath_rate_matrix(d, T, eta=ETA_DEFAULT, K=N_BATH_STATES):
    """Q[j, i] = rate (per atomic time unit) of hopping from level i to level j; columns sum to zero."""
    E, X, kT = d["E"][:K], d["X"][:K, :K], KB_AU * T
    w = E[:, None] - E[None, :]                              # w[i, j] = E_i - E_j
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        n = 1.0 / np.expm1(np.where(w > 0, w, 1.0) / kT)     # thermal occupation (only used for w > 0)
    g = 2 * np.pi * eta * np.abs(w) * np.abs(X) ** 2
    down = np.where(w > 0, g * (n + 1), 0.0)                 # down[i, j]: i -> j, with E_i > E_j
    up = np.where(w > 0, g * n, 0.0)                         # up[i, j]:   j -> i
    R = down.T + up                                          # R[j, i] = rate i -> j
    return R - np.diag(R.sum(axis=0))


def mean_first_passage_rate(Q, target, start):
    """1 / (mean time to first reach any level in `target`, starting from level `start`)."""
    transient = ~target
    tau = solve(-Q[np.ix_(transient, transient)], np.ones(transient.sum()))
    return 1.0 / tau[np.cumsum(transient)[start] - 1]


def tautomerization_rates(d, T, eta=ETA_DEFAULT, K=N_BATH_STATES):
    """(forward rate canonical -> tautomer, reverse rate tautomer -> canonical) in 1/s."""
    Q = bath_rate_matrix(d, T, eta, K)
    taut = d["w"][:K] > 0.5
    start_f = 0                                              # canonical ground state
    start_r = int(np.argmax(taut))                           # lowest tautomer-side level
    k_f = mean_first_passage_rate(Q, taut, start_f)
    k_r = mean_first_passage_rate(Q, ~taut, start_r)
    to_per_s = 1.0 / (AU_TIME_FS * 1e-15)
    return k_f * to_per_s, k_r * to_per_s


def eyring_rate(Ef_eV, T):
    """Classical transition-state-theory rate (1/s) over a barrier Ef_eV at temperature T."""
    return (KB_EV * T / H_EV_S) * np.exp(-Ef_eV / (KB_EV * T))


def relaxation_curve(d, T, times_au, eta=ETA_DEFAULT, K=N_BATH_STATES):
    """P(tautomer side) vs time when the proton starts in its canonical ground state in a bath at T."""
    Q = bath_rate_matrix(d, T, eta, K)
    pi = np.exp(-(d["E"][:K] - d["E"][0]) / (KB_AU * T)); pi /= pi.sum()
    S = (Q * np.sqrt(pi)[None, :]) / np.sqrt(pi)[:, None]     # symmetrised generator (detailed balance)
    S = 0.5 * (S + S.T)
    lam, V = np.linalg.eigh(S)
    p0 = np.zeros(K); p0[0] = 1.0
    c = V.T @ (p0 / np.sqrt(pi))
    return np.array([(np.sqrt(pi) * (V @ (np.exp(lam * t) * c))) @ d["w"][:K] for t in times_au])


def analyze_base_pair(base_pair, temperature_K=310.0, eta=ETA_DEFAULT, n_qubits=6):
    """THE BRIDGE: biologist-facing entry point. base_pair in {'A-T', 'G-C'} -> tautomer propensity.
    Hides the grid, the Hamiltonian, the qubit mapping and the bath model."""
    d = eigen_data(base_pair, n_qubits)
    k_f, k_r = tautomerization_rates(d, temperature_K, eta)
    Ef = PRESETS[base_pair][1][0]
    return dict(base_pair=base_pair, temperature_K=temperature_K,
                tautomer_population=gibbs_tautomer_population(d, temperature_K),
                forward_rate_per_s=k_f, canonical_lifetime_s=1 / k_f,
                reverse_rate_per_s=k_r, tautomer_lifetime_s=1 / k_r,
                classical_forward_rate_per_s=eyring_rate(Ef, temperature_K))


def day5_open_system(eta=ETA_DEFAULT):
    print("\n=== DAY 5: open quantum system (thermal bath) ===")
    names = ["G-C", "A-T", "G-C (Morse 2022)"]
    data = {n: eigen_data(n) for n in names}
    temps = np.linspace(250, 400, 31)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    # --- 1. equilibrium tautomer population vs temperature ---
    print("Equilibrium tautomer-side population (thermal equilibrium of the bath-coupled proton):")
    for n, c in zip(names, ["tab:blue", "tab:red", "tab:green"]):
        pops = [gibbs_tautomer_population(data[n], T) for T in temps]
        ax[0].semilogy(temps, pops, color=c, label=n)
        print(f"  {n:<18} 300 K: {gibbs_tautomer_population(data[n], 300):.2e}   "
              f"310 K: {gibbs_tautomer_population(data[n], 310):.2e}")
    ax[0].set_xlabel("temperature (K)"); ax[0].set_ylabel("P(tautomer side) at equilibrium")
    ax[0].set_title("Equilibrium tautomer population"); ax[0].legend(fontsize=8)

    # --- 2. rates vs temperature: bath model vs classical over-the-barrier ---
    print("\nForward (canonical -> tautomer) and reverse rates at 298.15 K, with published lifetimes:")
    for n, c in [("G-C", "tab:blue"), ("A-T", "tab:red")]:
        rates = np.array([tautomerization_rates(data[n], T, eta)[0] for T in temps])
        classical = np.array([eyring_rate(PRESETS[n][1][0], T) for T in temps])
        ax[1].semilogy(1000 / temps, rates, "-", color=c, label=f"{n} bath model")
        ax[1].semilogy(1000 / temps, classical, "--", color=c, label=f"{n} classical TST")
        k_f, k_r = tautomerization_rates(data[n], 298.15, eta)
        lit = LIT_LIFETIMES[n]
        print(f"  {n}: tau_forward = {1/k_f:.1e} s   (published {lit['scattering'][0]:.1e} - {lit['WKB'][0]:.1e} s)")
        print(f"       tau_reverse = {1/k_r:.1e} s   (published {lit['scattering'][1]:.1e} - {lit['WKB'][1]:.1e} s)")
    ax[1].set_xlabel("1000 / T  (1/K)"); ax[1].set_ylabel("forward rate (1/s)")
    ax[1].set_title("Tautomerization rate vs temperature"); ax[1].legend(fontsize=8)

    # --- 3. relaxation toward equilibrium ---
    times = np.logspace(2, 17, 200)                                 # atomic time units
    for n, c in [("G-C", "tab:blue"), ("A-T", "tab:red")]:
        p = relaxation_curve(data[n], 310.0, times, eta)
        ax[2].loglog(times * AU_TIME_FS * 1e-15, np.maximum(p, 1e-16), color=c, label=n)
        ax[2].axhline(gibbs_tautomer_population(data[n], 310.0), color=c, ls=":", lw=1)
    ax[2].set_xlabel("time (s)"); ax[2].set_ylabel("P(tautomer side)"); ax[2].set_ylim(1e-14, 1e-4)
    ax[2].set_title("Proton starting in canonical state, 310 K"); ax[2].legend(fontsize=8)
    plt.tight_layout(); plt.savefig("day5_open_system.png", dpi=120)
    print("Saved plot -> day5_open_system.png")

    # --- 4. the biologist-facing bridge ---
    print("\nanalyze_base_pair(...) at 310 K:")
    for bp in ["A-T", "G-C"]:
        r = analyze_base_pair(bp, 310.0, eta)
        print(f"  {bp}: tautomer population {r['tautomer_population']:.2e}, "
              f"canonical lifetime {r['canonical_lifetime_s']:.1e} s, tautomer lifetime {r['tautomer_lifetime_s']:.1e} s")
    plt.show()


if __name__ == "__main__":
    # python main.py          -> runs Days 1-5.     python main.py 4 5  -> runs only Days 4 and 5
    days = {int(a) for a in sys.argv[1:]} or {1, 2, 3, 4, 5}
    if 1 in days: day1_qiskit_basics()
    if 2 in days: day2_double_well()
    if 3 in days: day3_trotter()
    if 4 in days: day4_real_potentials()
    if 5 in days: day5_open_system()