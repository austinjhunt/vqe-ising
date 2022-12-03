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
