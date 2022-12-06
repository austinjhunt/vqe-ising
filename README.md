# Implementing the Variational Quantum Eigensolver (VQE) Algorithm and Quantum Approximate Optimization Algorithm (OAOA) Using [Qiskit](https://qiskit.org)

This is a final project for [CS 8395 - Special Topics: Introduction to Quantum Computing]() at [Vanderbilt University](https://vanderbilt.edu) taught by [Dr. Chuck Easttom](http://www.chuckeasttom.com/)

## Problem Statement

Choose an algorithm we have not covered and implement it in the programming language of your choice, along with a paper describing the algorithm, and explaining your choice of algorithm and programming language.

## Variational Quantum Eigensolver (VQE)

Introduced by Peruzzo et al. in a July 2014 in their paper [arXiv:1304.3061](https://arxiv.org/abs/1304.3061), VQE is a hybrid quantum-classical algorithm for efficiently finding eigenvalues of eigenvectors in large problem spaces that was first implemented with a combination of a small-scale photonic quantum processor (quantum computing) and a conventional computer (classical computing). As stated in the [Qiskit documentation](https://qiskit.org/documentation/stubs/qiskit.algorithms.VQE.html#qiskit.algorithms.VQE), the algorithm specifically leverages a **"variational technique (discussed below) to find the minimum eigenvalue of the Hamiltonian _H_ of a given system"**.

### Variational?

The [variational method](<https://en.wikipedia.org/wiki/Variational_method_(quantum_mechanics)>) in quantum mechanics is "a way of approximating the lowest-energy eigenstate or ground state, and some excited states." The method consists of choosing a "trial wavefunction" depending on one or more parameters, and finding the values of these parameters for which the expectation value of the energy is the lowest possible. The wavefunction obtained by fixing the parameters to such values is then an approximation to the ground state wavefunction, and the expectation value of the energy in that state is an upper bound to the ground state energy

#### Ground States vs Excited States in Quantum Systems

The ground state of a quantum-mechanical system is its stationary state of lowest energy; the energy of this state is also known as the zero-point energy of the system. An excited state is any state with energy greater than the ground state.

## Quantum Approximate Optimization Algorithm (QAOA)

Introduced by Edward Farhi, Jeffrey Goldstone, and Sam Gutmann in their paper [arXiv:1411.4028](https://arxiv.org/abs/1411.4028), QAOA is a quantum algorithm that produces approximate solutions for combinatorial optimization problems. As stated in the abstract, _"The algorithm depends on a positive integer p and the quality of the approximation improves as p is increased. The quantum circuit that implements the algorithm consists of unitary gates whose locality is at most the locality of the objective function whose optimum is sought. The depth of the circuit grows linearly with p times (at worst) the number of constraints. If p is fixed, that is, independent of the input size, the algorithm makes use of efficient classical preprocessing. If p grows with the input size a different strategy is proposed."_



# Ising model Background for Cirq Implementation in [sample.py](sample.py)
Suppose you have a system of N atoms on a square lattice (grid). So the length and width of the grid is sqrt(N). 
Each atom can be in one of two possible states, e.g., spin up or spin down. 

Any two neighboring atoms are coupled by an interaction -J. A positive value for J favors parallel alignment of neighboring atoms (e.g., up up or down down). In other words, $J > 0 \implies \uparrow \uparrow \text{ or } \downarrow \downarrow, J < 0 \implies \uparrow \downarrow \text{ or } \downarrow \uparrow$. We can also think of a term to manipulate each atom individually. In the case of localized spins, this can be achieved by an external magnetic field $B$. In the presense of a magnetic field $B$, the magnetic moments will tend to align. 

This model can be described in terms of the following Hamiltonian: 

$$H = \frac{gs}{2} \mu_B B \sum_{i=1}^{N} \sigma_z^i - J \sum_{<i,j>}\sigma_z^i \sigma_z^j$$

where the first term represents the system coupling to the external magnetic field $B$, $gs$ is the spin-g factor, and $\mu_B$ is the Bohr magneton, and the second term represents coupling of neighboring atoms, a sum of all nearest neighbors, and the product of sigma matrices at atom $i$ and $j$. 

Note that the Hilbert space is spanned by all possible combinations of $\uparrow$ / $\downarrow$ at each atom. Therefore, the Pauli matrix $\sigma_z^i$ reads:

$$\sigma_z^i = | \uparrow \rangle_i \langle_i \uparrow | - | \downarrow \rangle_i \langle_i \downarrow |  $$

Another resource actually ignores the external magnetic field entirely and describes the total energy of the system as 

$$E_\mu = \sum_{<i,j>} -J \sigma_i \sigma_j =  -J \sum_{<i,j>} \sigma_i \sigma_j =$$

which only captures the coupling of neighboring atoms. That is, take each atom, get the spin direction of the atom to the up/down/left/right, sum that total, and add that to a global total for the full grid. 