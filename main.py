from ast import operator
import numpy as np
import pylab
import copy

from qiskit import BasicAer  # for QC simulator
from qiskit.utils import QuantumInstance  # for QC simulator
from qiskit.algorithms import (
    # tool that will give exact energies based on a classical calculation
    NumPyMinimumEigensolver,
    VQE  # to initialize the Variational Quantum Eigensolver (VQE) algorithm
)
# classical optimizer to help update the ansatz
from qiskit.algorithms.optimizers import SLSQP
from qiskit_nature.circuit.library import (
    UCCSD,  # tool that will help vary the HartreeFock guess into the VQE ansatz
    HartreeFock  # initial ansatz which we need to optimize using VQE
)
# will help set up molecule
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
# hamiltonian operator representing total energy of quantum system
from qiskit_nature.second_q.hamiltonians import Hamiltonian
# will help to do the mapping (encoding molecule information to qubits)
from qiskit_nature.second_q.mappers import ParityMapper


def setup_molecule():
    """ Setup Lithium Hydride molecule model """
    # H = Hydrogen. Distances in 3D where H atom is going to exist.
    # Curly braces represent dimension where distance can vary. We will vary distance of lithium hydride in the z direction.
    molecule = 'H .0 .0 -{0}; Li .0 .0 {0}'

    # specify distances at which we want to calculate energies.
    # from 0.5 angstroms to 4 angstroms at intervals of 0.25 angstroms, where an angstrom is just 10^-10 meters
    distances = np.arange(0.5, 4.25, 0.25)
    # array to store energies. Ground state energies calculated by VQE
    vqe_energies = []
    # Energies from initial hartreefock guess. (initial guess that has not been optimized by VQE)
    hf_energies = []

    # exact energies calculated with the numpy minimum eigensolver.
    exact_energies = []

    # loop over the distances and compute the VQE
    for i, d in enumerate(distances):
        print(f'Step={i}, distance={d}angstroms')

        # set up experiment
        # main part of experiment is driver
        # vary the disatance & compute energy at each distance;
        # basis indicates how driver will represent electronic orbitals
        driver = PySCFDriver(
            atom=molecule.format(d/2),
            basis='sto3g',
            unit=DistanceUnit.ANGSTROM
        )

        # set up quantum molecule.
        # setting up for classical simulation.
        # As you increase size of molecule, takes longer and longer for classical computer to run.
        # Running this driver, will yield an ElectronicStructureProblem, Qiskit Nature’s representation of the electronic structure problem which we are interested in solving.
        qmolecule = driver.run()

        # hamiltonian operator represents energy of the system
        # qubit mapping is how you encode information of molecule into the qubits of the quantum computer
        # hamiltonian_operator = Hamiltonian(
        #     qubit_mapping=ParityMapper(),  # use PARITY mapping type
        #     # trick to speed up calculation (Principal Component Analysis?)
        #     two_qubit_reduction=True,
        #     # freeze the orbitals within the molecule in the core that don't contribute to bonding
        #     freeze_core=True,
        #     # reduce the orbitals that don't contribute to calculation either. Known ahead of time
        #     orbital_reduction=[-3, -2]
        # )
        hamiltonian_operator = qmolecule.hamiltonian

        # use operator to run classical calculation
        qubit_op, aux_ops = hamiltonian_operator.run(qmolecule=qmolecule)

        # get the exact classical result from the numpy minimum eigensolver
        # (so we can compare VQE results with exact classical results)
        exact_result = NumPyMinimumEigensolver(
            qubit_op,
            aux_operators=aux_ops
        )
        # Process result
        exact_result = hamiltonian_operator.process_algorithm_result(
            exact_result)

        # VQE portion

        # define optimizer
        # try 1000 times before converging to local minimum
        optimizer = SLSQP(maxiter=1000)
        # the initial state will be the hartreefock state
        initial_state = HartreeFock(
            hamiltonian_operator.molecule_info['num_orbitals'],
            hamiltonian_operator.molecule_info['num_particles'],
            qubit_mapping=hamiltonian_operator._qubit_mapping,
            two_qubit_reduction=hamiltonian_operator._two_qubit_reduction,
        )

        # set up variational form/ method so we can take that initial state & do the variations
        # to find minimum energy (ground state)
        # variational form is a parameterized circuit with a fixed form
        var_form = UCCSD(
            num_orbitals=hamiltonian_operator.molecule_info['num_orbitals'],
            num_particles=hamiltonian_operator.molecule_info['num_particles'],
            initial_state=initial_state,  # the initial state defined above
            qubit_mapping=hamiltonian_operator._qubit_mapping,
            two_qubit_reduction=hamiltonian_operator._two_qubit_reduction,
        )

        # Initialize the Variational Quantum Eigensolver algorithm
        algorithm = VQE(
            operator=qubit_op,
            var_form=var_form,
            optimizer=optimizer,
            aux_operators=aux_ops
        )

        # Now want to run the algorithm and obtain the results
        # Use the Quantum Instance to run it with BasicAer using the statevector_simulator backend
        vqe_result = algorithm.run(
            QuantumInstance(BasicAer.get_backend('statevector_simulator'))
        )
        # process the vqe result
        vqe_result = hamiltonian_operator.process_algorithm_result(vqe_result)

        # from this vqe result, we can get all of our energies
        exact_energies.append(exact_result.energy)
        vqe_energies.append(vqe_result.energy)
        hf_energies.append(vqe_result.hartree_fock_energy)


if __name__ == "__main__":
    setup_molecule()
