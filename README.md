# Proton Tunneling in DNA: a Qiskit bridge for biologists

> **Hackathon challenge:** use Qiskit to create a new tool, interface, or abstraction that bridges quantum computing and a non-quantum field.
> **Our answer:** a workflow that takes a DNA base pair (or strand) and a temperature and returns how often, and for how long, the pair sits in its rare *tautomer* form. The quantum work (qubit mapping, time-evolution circuits, a shallow variational circuit) happens behind one function call and a web app, so a biologist never sees a gate.

**Authors:** _Anaisha Jain_  **Hackathon:** _IBM Qiskit Hackathon_

#### Check it out! https://qiskithackathon-nanobots.streamlit.app/

---

## 1. Motivation

DNA replicates accurately because each base pair holds its hydrogen-bonded protons in the right places. Occasionally a proton can **tunnel** across the energy barrier to the neighbouring base and form a rare *tautomer*. If a tautomer exists at the moment DNA is copied, it can be mispaired, and this has been proposed as one source of spontaneous point mutations.

Two things make this a good fit for a quantum-computing bridge:

- The proton is a quantum particle, so the physics is quantum mechanical. Larger versions of the problem (more protons, more particles, realistic environments) are the kind that grow exponentially for classical methods.
- The people who care about the answer are biologists and biochemists, who should not have to learn gates, Pauli operators or Trotterization to ask the question.

> **Important scope note.** This tool models only **step 1** of a long chain: *tautomer exists → is present at replication → gets mispaired → escapes repair → lands somewhere harmful → contributes to cancer.* It does **not** predict mutations or cancer risk. Whether proton tunneling really drives spontaneous mutation is still debated in the literature.

## 2. What it does

| Input | Output |
|---|---|
| A base pair (A-T or G-C), or a DNA strand, plus a temperature | Tautomer population, tautomer lifetime, canonical lifetime, and a per-position propensity along the strand |

The same pipeline sits behind all of these entry points:

```python
analyze_base_pair("G-C", temperature_K=310)   # one pair type
analyze_strand("ATGCCGTAGC...", 310)          # a whole strand
```

and a Streamlit web app (`streamlit run main.py`) with three tabs: **Base pair**, **DNA strand**, and **How it works & limits**.

## 3. How it works (the "bridge")

```
base pair + temperature
        |
        v
 published proton-transfer potential (atomic units)        <- physics
        |
        v
 64-point grid Hamiltonian  ->  SparsePauliOp (6 qubits)   <- Qiskit mapping
        |
        +--> Trotter circuits (PauliEvolutionGate)  : coherent dynamics, checked against exact
        +--> shallow variational circuit (VQD)      : lowest energy levels, hardware-friendly
        |
        v
 thermal bath (weak-coupling master equation)              <- turns levels into populations/rates
        |
        v
 tautomer population, lifetimes, per-position propensity
```

### Qiskit components used

| Component | Role |
|---|---|
| `SparsePauliOp.from_operator` | Converts the 64×64 physical Hamiltonian into a 6-qubit operator (about 120 Pauli terms) |
| `PauliEvolutionGate` with `LieTrotter` / `SuzukiTrotter` | Builds the time-evolution circuits |
| `Statevector`, `Operator` | Exact simulation of circuits and validation against classical results |
| `RealAmplitudes` ansatz + VQD | Finds the lowest energy levels with a depth-16 circuit |
| `generate_preset_pass_manager`, `EstimatorV2` (`qiskit-ibm-runtime`) | Transpilation and noisy-chip evaluation (`ibm_hardware.py`) |

### Physics inputs (from the literature)

- A-T and G-C proton-transfer barriers (forward / reverse, eV): **A-T 0.581 / 0.011**, **G-C 0.693 / 0.266** from Slocombe, Al-Khalili & Sacchi (2021), fitted with a smooth asymmetric double well.
- A **G-C Morse potential** from Slocombe, Sacchi & Al-Khalili (2022). Our implementation reproduces its published barriers (0.705 / 0.270 eV, asymmetry 0.435 eV).
- The code prints a model-vs-literature check, and all three potentials match.

## 4. Results

### Validation

| Check | Result |
|---|---|
| Potentials vs published barriers | Exact match |
| 6-qubit grid vs finer grids | Low energy levels within about 1.4% of 8 qubits |
| Trotter circuits vs exact dynamics | Order-4 Suzuki, 1500 steps: max error about 2×10⁻⁴ |
| Variational circuit vs exact diagonalization | Level errors of 0.05 meV, 0.56 meV and 2.8 meV for levels 0-2; the first level spacing (about 100 meV) is off by 0.5 meV |

### Physics at 310 K

| | A-T | G-C |
|---|---|---|
| Tautomer population | ~1.3×10⁻⁹ | ~1.6×10⁻⁷ |
| Tautomer lifetime | ~2×10⁻¹³ s | ~6×10⁻⁹ s |

G-C is about **120× more likely** than A-T to be found in its tautomer form and holds it about **30,000× longer**, in line with the qualitative picture in the literature.

### Circuit cost (6 qubits)

| Circuit | Depth | CNOTs |
|---|---|---|
| Variational ansatz (VQD) | 16 | 20 |
| One Lie-Trotter step | 1,129 | 708 |
| One order-4 Trotter step (about 1,500 needed) | 11,255 | 7,050 |

Measuring the energy on a device takes about 33 measurement circuits per evaluation. The variational circuit is hundreds of times shallower than a single Trotter step, which is why it is the hardware-friendly part.

## 5. Limitations (please read)

- **No claim of quantum advantage.** At 6 qubits a laptop solves this exactly. The contribution is a validated, extensible bridge.
- **Weak-coupling bath model.** Equilibrium tautomer populations are far below the 1.7×10⁻⁴ reported for G-C with a strong-coupling treatment, and our forward (canonical → tautomer) lifetimes are 1-2 orders of magnitude slower than published tunneling-corrected values. Our A-T tautomer lifetime (about 2×10⁻¹³ s) is in the published range; our G-C tautomer lifetime (about 9×10⁻⁹ s at 298 K) is 1-2 orders of magnitude longer than published. Read results as **relative propensity**, not absolute predictions. The bath-strength parameter is illustrative.
- **1D model.** No stacking or neighbour effects, no protein or solvent effects, and a single reaction coordinate. In strand mode, every A/T gives the same value and every G/C gives the same value.
- **A-T well positions are borrowed** from the G-C model (the literature barriers fix only the energies).
- **Trotter circuits are far too deep for current hardware** (see the table above).
- **Hardware / noise.** We ran the variational circuit through a **noisy simulator of an IBM chip** (`python ibm_hardware.py --fake`, FakeTorino noise model). Absolute energy errors were about 220-290 meV, larger than the 100 meV level spacing, and error mitigation barely helped. The Hamiltonian has many large Pauli terms, so a few percent of noise swamps the signal. We did **not** report a real-hardware result. `ibm_hardware.py` is included so the run can be reproduced on a real backend.
- **Mechanism is debated.** A toy model cannot settle whether tunneling causes mutations.

## 6. Running it

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

| Command | What it does |
|---|---|
| `streamlit run main.py` | Launches the web app |
| `python main.py` | Runs Days 1-7 in order (plot windows open one at a time; Day 6 takes about a minute) |
| `python main.py 4 5` | Runs only the chosen days |
| `python ibm_hardware.py --fake` | Noisy-simulator run of the variational circuit (free) |
| `python ibm_hardware.py` | Real IBM hardware (needs a saved IBM Quantum account) |

`requirements.txt` lists `qiskit`, `numpy`, `scipy`, `matplotlib` and `streamlit`. The hardware script additionally needs `qiskit-ibm-runtime` (and `qiskit-aer` for `--fake`).

### What each part of `main.py` does

| Day | Content |
|---|---|
| 1 | Qiskit basics: circuits, `SparsePauliOp`, Trotter vs exact on a 2-qubit toy |
| 2 | Classical double-well model with dimensionless units (ground truth) |
| 3 | Quantum version: Hamiltonian → Pauli terms, Trotter circuits vs exact, scaling table |
| 4 | Real A-T / G-C potentials in atomic units, validated against the literature |
| 5 | Thermal bath: populations, rates and lifetimes vs temperature; `analyze_base_pair` |
| 6 | Variational quantum deflation: shallow circuit for the lowest energy levels |
| 7 | DNA strand mode and the Streamlit app |

## 7. Future work

- Strong-coupling open-system dynamics (e.g. Caldeira-Leggett-type treatments) to close the gap with the published rates.
- Two coupled protons (A-T has two hydrogen bonds, G-C has three), where the Hamiltonian and the classical cost grow.
- Neighbour (stacking) effects using electronic-structure inputs.
- Better qubit encodings and operator reductions so the Hamiltonian is less sensitive to noise, then real-hardware runs with stronger error mitigation.

## 8. References

1. L. Slocombe, M. Sacchi, J. Al-Khalili, *An open quantum systems approach to proton tunnelling in DNA*, Communications Physics 5, 109 (2022).
2. L. Slocombe, J. Al-Khalili, M. Sacchi, *Quantum and classical effects in DNA point mutations: Watson-Crick tautomerism in AT and GC base pairs*, Physical Chemistry Chemical Physics 23, 4141 (2021).
3. Qiskit SDK and Qiskit Runtime documentation, IBM Quantum.
