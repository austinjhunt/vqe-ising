"""
https://github.com/Qiskit/qiskit-tutorials
https://qiskit.org/textbook/ch-applications/vqe-molecules.html
"""
import matplotlib.pyplot as plt
import numpy as np

from qiskit.algorithms import VQE
from qiskit_nature.second_q.algorithms import GroundStateSolver, NumPyMinimumEigensolverFactory
from qiskit_nature.drivers import Molecule

from qiskit_nature.drivers.second_quantization import (
    ElectronicStructureMoleculeDriver, ElectronicStructureDriverType)
from qiskit_nature.transformers.second_quantization.electronic import FreezeCoreTransformer
# Deprecated
# from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import ParityMapper

from qiskit_nature.circuit.library import UCCSD, HartreeFock
from qiskit.circuit.library import EfficientSU2
from qiskit.algorithms.optimizers import COBYLA, SPSA, SLSQP
from qiskit.opflow import TwoQubitReduction
from qiskit import BasicAer, Aer
from qiskit.utils import QuantumInstance
from qiskit.utils.mitigation import CompleteMeasFitter
from qiskit.providers.aer.noise import NoiseModel


"""
Running VQE on the statevector Simulator

We demonstrate the calculation of the ground state
energy for LiH at various interatomic distances.
A driver for the molecule must be created at each
such distance. Note that in this experiment, to
reduce the number of qubits used, we freeze the core
and remove two unoccupied orbitals. First, we define
a function that takes an interatomic distance and
returns the appropriate qubit operator, H
, as well as some other information about the operator.
"""


def get_qubit_op(dist):
    # Define Molecule
    molecule = Molecule(
        # Coordinates in Angstrom
        geometry=[
            ["Li", [0.0, 0.0, 0.0]],
            ["H", [dist, 0.0, 0.0]]
        ],
        multiplicity=1,  # = 2*spin + 1
        charge=0,
    )

    driver = ElectronicStructureMoleculeDriver(
        molecule=molecule,
        basis="sto3g",
        driver_type=ElectronicStructureDriverType.PYSCF)

    # Get properties
    properties = driver.run()
    num_particles = (properties
                     .get_property("ParticleNumber")
                     .num_particles)
    num_spin_orbitals = int(properties
                            .get_property("ParticleNumber")
                            .num_spin_orbitals)

    # Define Problem, Use freeze core approximation, remove orbitals.
    problem = ElectronicStructureProblem(
        driver,
        [FreezeCoreTransformer(freeze_core=True,
                               remove_orbitals=[-3, -2])])

    second_q_ops = problem.second_q_ops()  # Get 2nd Quant OP
    num_spin_orbitals = problem.num_spin_orbitals
    num_particles = problem.num_particles

    mapper = ParityMapper()  # Set Mapper
    hamiltonian = second_q_ops[0]  # Set Hamiltonian
    # Do two qubit reduction
    converter = QubitConverter(mapper, two_qubit_reduction=True)
    reducer = TwoQubitReduction(num_particles)
    qubit_op = converter.convert(hamiltonian)
    qubit_op = reducer.convert(qubit_op)

    return qubit_op, num_particles, num_spin_orbitals, problem, converter


"""
First, the exact ground state energy is calculated using the qubit operator and a classical exact eigensolver. Subsequently, the initial state |ψ⟩ is created, which the VQE instance uses to produce the final ansatz minθ(|ψ(θ)⟩). The exact result and the VQE result at each interatomic distance is stored. Observe that the result given by vqe.run(backend)['energy'] + shift is equivalent the quantity minθ(⟨ψ(θ)|H|ψ(θ)⟩)

, where the minimum is not necessarily the global minimum.

When initializing the VQE instance with VQE(qubit_op, var_form, optimizer, 'matrix') the expectation value of H
on |ψ(θ)⟩ is directly calculated through matrix multiplication. However, when using an actual quantum device, or a true simulator such as the qasm_simulator with VQE(qubit_op, var_form, optimizer, 'paulis') the calculation of the expectation value is more complicated. A Hamiltonian may be represented as a sum of a Pauli strings, with each Pauli term acting on a qubit as specified by the mapping being used. Each Pauli string has a corresponding circuit appended to the circuit corresponding to |ψ(θ)⟩. Subsequently, each of these circuits is executed, and all of the results are used to determine the expectation value of H on |ψ(θ)⟩

. In the following example, we initialize the VQE instance with matrix mode, and so the expectation value is directly calculated through matrix multiplication.

Note that the following code snippet may take a few minutes to run to completion.

"""


def exact_solver(problem, converter):
    solver = NumPyMinimumEigensolverFactory()
    calc = GroundStateEigensolver(converter, solver)
    result = calc.solve(problem)
    return result


backend = BasicAer.get_backend("statevector_simulator")
distances = np.arange(0.5, 4.0, 0.2)
exact_energies = []
vqe_energies = []
optimizer = SLSQP(maxiter=5)

# pylint: disable=undefined-loop-variable
for dist in distances:
    (qubit_op, num_particles, num_spin_orbitals,
     problem, converter) = get_qubit_op(dist)
    result = exact_solver(problem, converter)
    exact_energies.append(result.total_energies[0].real)
    init_state = HartreeFock(num_spin_orbitals, num_particles, converter)
    var_form = UCCSD(converter,
                     num_particles,
                     num_spin_orbitals,
                     initial_state=init_state)
    vqe = VQE(var_form, optimizer, quantum_instance=backend)
    vqe_calc = vqe.compute_minimum_eigenvalue(qubit_op)
    vqe_result = problem.interpret(vqe_calc).total_energies[0].real
    vqe_energies.append(vqe_result)
    print(f"Interatomic Distance: {np.round(dist, 2)}",
          f"VQE Result: {vqe_result:.5f}",
          f"Exact Energy: {exact_energies[-1]:.5f}")

print("All energies have been calculated")


# Plot results

plt.plot(distances, exact_energies, label="Exact Energy")
plt.plot(distances, vqe_energies, label="VQE Energy")
plt.xlabel('Atomic distance (Angstrom)')
plt.ylabel('Energy')
plt.legend()
plt.show()
