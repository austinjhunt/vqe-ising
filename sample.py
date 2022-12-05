""" This is a sample walkthrough of using the VQE algorithm 
to find the ground state of 2D +/- Ising model with transverse field

https://iopscience.iop.org/article/10.1088/0305-4470/15/10/028/meta
Abstract: 
In a spin glass with Ising spins, the problems of computing 
the magnetic partition function and finding a ground state are studied. 
In a finite two-dimensional lattice these problems can be solved by 
algorithms that require a number of steps bounded by a polynomial 
function of the size of the lattice. In contrast to this fact, 
the same problems are shown to belong to the class of NP-hard problems, 
both in the two-dimensional case within a magnetic field, and in the 
three-dimensional case. NP-hardness of a problem suggests that it is 
very unlikely that a polynomial algorithm could exist to solve it.

"""
import cirq

# Create a circuit on a Grid 
# ISING problem has a natural structure on a grid,
# so use Circ's built in cirq.GridQubit as our qubits

# define length and width of grid
length = 3 

# define qubits on the grid
qubits = cirq.GridQubit.square(length)
print(qubits)

# Here we see that we've created a bunch of cirq.GridQubits,
# which have a row and column, indicating their position on a grid.

# Now that we have our qubits, let's build a cirq.Circuit on these qubits.
# For example, suppose we want to apply the Hadamard gate cirq.H to every 
# qubit whose row index plus column index is even and an cirq.X gate to 
# every qubit whose row index plus column index is odd. 
# To do this we write 

# circuit = cirq.Circuit()
# circuit.append(cirq.H(q) for q in qubits if (q.row + q.col) % 2 == 0)
# circuit.append(cirq.X(q) for q in qubits if (q.row + q.col) % 2 != 0)
# print(circuit)


# Creating the Ansatz 
# one convenient pattern is to use a python Generator for defining sub-circuits or layers in our algorithm. 
# we will define a function that takes in the relevant parameters and then yields the operations 
# for the sub-circuit, then this can be appended to the cirq.Circuit

def rot_x_layer(length, half_turns):
    """ 
    Yields X rotations by half_turns on a square grid of a given length 
    """
    # Define the gate once and then re-use it for each Operation 
    rot = cirq.XPowGate(exponent=half_turns)

    # Create an X rotation Operation for each qubit in the grid 
    for i in range(length):
        for j in range(length):
            yield rot(cirq.GridQubit(i, j))


# Create the circuit using the rot_x_layer generator
circuit = cirq.Circuit() 
circuit.append(rot_x_layer(2, 0.1))
print(circuit)

# Note that rotation gate specified in half turns (ht)
# For a rotation about X axis, the gate is:
# cos(ht * pi)I + i*sin(ht*pi)X

# There is a lot of freedom defining a variational ansatz. 
# Here we will do a variation on a QAOA strategy and define 
# an ansatz related to the problem we are trying to solve.

# First we need to choose how the instances of the problem are 
# represented. These are the values J and h in the
# hamiltonian definition. 
# We represent them as 2D arrays (lists of lists). For J we use two such lists, one for 
# the row links and one for the column links. 

# Snippet for generating random problem instances. 

import random
def rand2d(rows, cols):
    return [[random.choice([+1, -1]) for _ in range(cols)] for _ in range(rows)]

def random_instance(length):
    # transverse field terms
    h = rand2d(length, length)
    # links within a row
    jr = rand2d(length - 1, length)
    # links within a column
    jc = rand2d(length, length - 1)
    return (h, jr, jc)

h, jr, jc = random_instance(3)
print(f'transverse fields: {h}')
print(f'row j fields: {jr}')
print(f'column j fields: {jc}')

"""
Ex output: 
transverse fields: [[1, 1, 1], [1, -1, 1], [-1, 1, -1]]
row j fields: [[1, 1, 1], [-1, -1, 1]]
column j fields: [[-1, -1], [-1, -1], [-1, -1]]
"""

# In the code above, the actual values
# will be different for each individual 
# run because they are using random.choice.


# Given this definition of the problem instance, 
# we can now introduce our ansatz. It will consist
# of one step of a circuit made up of:

# 1. Apply an initial mixing step that puts all qubits into the |+> = 1/sqrt(2) (|0>+|1>) state. (superposition...hadamard)
def prepare_plus_layer(length):
    for i in range(length):
        for j in range(length):
            yield cirq.H(cirq.GridQubit(i,j))

# 2. Apply a cirq.ZPowGate for the same parameter for all qubits where the transverse field term h is +1
def rot_z_layer(h, half_turns):
    """Yields Z rotations by half_turns conditioned on the field h."""
    gate = cirq.ZPowGate(exponent=half_turns)
    for i, h_row in enumerate(h):
        for j, h_ij in enumerate(h_row):
            if h_ij == 1:
                yield gate(cirq.GridQubit(i, j))


# 3. Apply a cirq.CZPowGate for the same parameter between all qubits where the coupling field term  is . If the field is , apply cirq.CZPowGate conjugated by  gates on all qubits.
def rot_11_layer(jr, jc, half_turns):
    """Yields rotations about |11> conditioned on the jr and jc fields."""
    cz_gate = cirq.CZPowGate(exponent=half_turns)    
    for i, jr_row in enumerate(jr):
        for j, jr_ij in enumerate(jr_row):
            q = cirq.GridQubit(i, j)
            q_1 = cirq.GridQubit(i + 1, j)
            if jr_ij == -1:
                yield cirq.X(q)
                yield cirq.X(q_1)
            yield cz_gate(q, q_1)
            if jr_ij == -1:
                yield cirq.X(q)
                yield cirq.X(q_1)

    for i, jc_row in enumerate(jc):
        for j, jc_ij in enumerate(jc_row):
            q = cirq.GridQubit(i, j)
            q_1 = cirq.GridQubit(i, j + 1)
            if jc_ij == -1:
                yield cirq.X(q)
                yield cirq.X(q_1)
            yield cz_gate(q, q_1)
            if jc_ij == -1:
                yield cirq.X(q)
                yield cirq.X(q_1)

# 4. Apply an cirq.XPowGate for the same parameter for all qubits. 
# This is the method rot_x_layer we have written above.

# Putting all together, we can create a step that uses just three parameters. 
# Below is the code, which uses the generator for each of the layers 
# (note to advanced Python users: this code does not contain a bug in 
# using yield due to the auto flattening of the OP_TREE concept. 
# Typically, one would want to use yield from here, but this is not necessary):

def initial_step(length):
    yield prepare_plus_layer(length)

def one_step(h, jr, jc, x_half_turns, h_half_turns, j_half_turns):
    length = len(h)
    yield rot_z_layer(h, h_half_turns)
    yield rot_11_layer(jr, jc, j_half_turns)
    yield rot_x_layer(length, x_half_turns)

h, jr, jc = random_instance(3)

circuit = cirq.Circuit()  
circuit.append(initial_step(len(h)))
circuit.append(one_step(h, jr, jc, 0.1, 0.2, 0.3))
# Here we see that we have chosen particular parameter values 0.1, 0.2, 0.3 
print(circuit)

# Simulation
# In Cirq, the simulators make a distinction between 
# a run and a simulation. A run only allows for a simulation 
# that mimics the actual quantum hardware. For example, it does 
# not allow for access to the amplitudes of the wave function 
# of the system, since that is not experimentally accessible. 
# Simulate commands, however, are broader and allow different 
# forms of simulation. When prototyping small circuits, it is 
# useful to execute simulate methods, but one should be wary 
# of relying on them when running against actual hardware.

simulator = cirq.Simulator()
circuit = cirq.Circuit()
circuit.append(initial_step(len(h)))
circuit.append(one_step(h, jr, jc, 0.1, 0.2, 0.3))
circuit.append(cirq.measure(*qubits, key='x'))
results = simulator.run(circuit, repetitions=100)
print(results.histogram(key='x'))

# Example output: 
# Counter({275: 5, 343: 2, 355: 2, 471: 2, 135: 2, 206: 2, 
#  218: 2, 133: 2, 302: 2, 83: 2, 279: 2, 291: 2, 387: 2, 
# 333: 2, 163: 2, 299: 1, 31: 1, 186: 1, 145: 1, 36: 1, 
# 312: 1, 269: 1, 112: 1, 67: 1, 377: 1, 398: 1, 407: 1, 
# 278: 1, 157: 1, 104: 1, 77: 1, 405: 1, 100: 1, 342: 1, 
# 287: 1, 28: 1, 265: 1, 309: 1, 34: 1, 481: 1, 21: 1, 
# 399: 1, 221: 1, 251: 1, 111: 1, 181: 1, 394: 1, 277: 1, 
# 416: 1, 169: 1, 161: 1, 198: 1, 317: 1, 263: 1, 357: 1, 
# 267: 1, 426: 1, 270: 1, 51: 1, 261: 1, 58: 1, 41: 1, 469: 1, 
# 268: 1, 489: 1, 43: 1, 470: 1, 167: 1, 475: 1, 103: 1, 
# 180: 1, 396: 1, 110: 1, 499: 1, 9: 1, 352: 1, 371: 1, 
# 307: 1, 185: 1, 273: 1, 354: 1, 395: 1})


# Note that we have run the simulation 100 times and produced 
# a histogram of the counts of the measurement results. 
# What are the keys in the histogram counter? Note that 
# we have passed in the order of the qubits. This ordering 
# is then used to translate the order of the measurement 
# results to a register using a big endian representation.

# For our optimization problem, we want to calculate the 
# value of the objective function for a given result run.
# One way to do this is using the raw measurement data 
# from the result of simulator.run. Another way to do 
# this is to provide to the histogram a method to 
# calculate the objective: this will then be used 
# as the key for the returned Counter.

import numpy as np

def energy_func(length, h, jr, jc):
    def energy(measurements):
        # Reshape measurement into array that matches grid shape.
        meas_list_of_lists = [measurements[i * length:(i + 1) * length]
                              for i in range(length)]
        # Convert true/false to +1/-1.
        pm_meas = 1 - 2 * np.array(meas_list_of_lists).astype(np.int32)

        tot_energy = np.sum(pm_meas * h)
        for i, jr_row in enumerate(jr):
            for j, jr_ij in enumerate(jr_row):
                tot_energy += jr_ij * pm_meas[i, j] * pm_meas[i + 1, j]
        for i, jc_row in enumerate(jc):
            for j, jc_ij in enumerate(jc_row):
                tot_energy += jc_ij * pm_meas[i, j] * pm_meas[i, j + 1]
        return tot_energy
    return energy
print(results.histogram(key='x', fold_func=energy_func(3, h, jr, jc)))

# Example output:
# Counter({5: 24, -1: 15, 1: 11, -3: 11, 7: 10, 3: 10, -5: 7, 
# 9: 5, 11: 3, 13: 2, -7: 1, -11: 1})

# One can then calculate the expectation value over all repetitions:
def obj_func(result):
    energy_hist = result.histogram(key='x', fold_func=energy_func(3, h, jr, jc))
    return np.sum([k * v for k,v in energy_hist.items()]) / result.repetitions
print(f'Value of the objective function {obj_func(results)}')

# Example output: Value of the objective function 2.34


# Parameterizing the Ansatz
# Now that we have constructed a variational ansatz and shown how 
# to simulate it using Cirq, we can think about optimizing the value.

# On quantum hardware, one would most likely want to 
# have the optimization code as close to the hardware 
# as possible. As the classical hardware that is 
# allowed to inter-operate with the quantum hardware 
# becomes better specified, this language will be better defined. 
# Without this specification, however, Cirq also provides a 
# useful concept for optimizing the looping in many 
# optimization algorithms. This is the fact that many 
# of the value in the gate sets can, instead of being specified 
# by a float, be specified by a symply.Symbol, and this sympy.Symbol 
# can be substituted for a value specified at execution time.

# Luckily for us, we have written our code so that using 
# parameterized values is as simple as passing sympy.Symbol 
# objects where we previously passed float values.

import sympy
circuit = cirq.Circuit()
alpha = sympy.Symbol('alpha')
beta = sympy.Symbol('beta')
gamma = sympy.Symbol('gamma')
circuit.append(initial_step(len(h)))
circuit.append(one_step(h, jr, jc, alpha, beta, gamma))
circuit.append(cirq.measure(*qubits, key='x'))
print(circuit)

# Note now that the circuit's gates are parameterized.

# Parameters are specified at runtime using a cirq.ParamResolver,
# which is just a dictionary from Symbol keys to runtime values.

# For instance, the following resolves the parameters to 
# actual values in the circuit.
resolver = cirq.ParamResolver({'alpha': 0.1, 'beta': 0.3, 'gamma': 0.7})
resolved_circuit = cirq.resolve_parameters(circuit, resolver)


# Cirq also has the concept of a sweep. A sweep is a collection of 
# parameter resolvers. This runtime information is very useful when 
# one wants to run many circuits for many different parameter values.
# Sweeps can be created to specify values directly (this is one way 
# to get classical information into a circuit), or a variety of helper 
# methods. For example suppose we want to evaluate our circuit 
# over an equally spaced grid of parameter values (Grid Search used in ML)
# We can easily create this using cirq.LinSpace.


sweep = (cirq.Linspace(key='alpha', start=0.1, stop=0.9, length=5)
         * cirq.Linspace(key='beta', start=0.1, stop=0.9, length=5)
         * cirq.Linspace(key='gamma', start=0.1, stop=0.9, length=5))
results = simulator.run_sweep(circuit, params=sweep, repetitions=100)
for result in results:
    print(result.params.param_dict, obj_func(result))

# Example output: 

""" 
OrderedDict([('alpha', 0.1), ('beta', 0.1), ('gamma', 0.1)]) 0.82
OrderedDict([('alpha', 0.1), ('beta', 0.1), ('gamma', 0.30000000000000004)]) 0.56
OrderedDict([('alpha', 0.1), ('beta', 0.1), ('gamma', 0.5)]) -0.6
OrderedDict([('alpha', 0.1), ('beta', 0.1), ('gamma', 0.7000000000000001)]) 0.02
OrderedDict([('alpha', 0.1), ('beta', 0.1), ('gamma', 0.9)]) -0.28
OrderedDict([('alpha', 0.1), ('beta', 0.30000000000000004), ('gamma', 0.1)]) 1.86
OrderedDict([('alpha', 0.1), ('beta', 0.30000000000000004), ('gamma', 0.30000000000000004)]) 2.88
OrderedDict([('alpha', 0.1), ('beta', 0.30000000000000004), ('gamma', 0.5)]) 1.18
OrderedDict([('alpha', 0.1), ('beta', 0.30000000000000004), ('gamma', 0.7000000000000001)]) 0.32
OrderedDict([('alpha', 0.1), ('beta', 0.30000000000000004), ('gamma', 0.9)]) 0.62
OrderedDict([('alpha', 0.1), ('beta', 0.5), ('gamma', 0.1)]) 2.5
OrderedDict([('alpha', 0.1), ('beta', 0.5), ('gamma', 0.30000000000000004)]) 1.76
OrderedDict([('alpha', 0.1), ('beta', 0.5), ('gamma', 0.5)]) 1.24
OrderedDict([('alpha', 0.1), ('beta', 0.5), ('gamma', 0.7000000000000001)]) 0.0
OrderedDict([('alpha', 0.1), ('beta', 0.5), ('gamma', 0.9)]) 0.48
OrderedDict([('alpha', 0.1), ('beta', 0.7000000000000001), ('gamma', 0.1)]) 1.94
OrderedDict([('alpha', 0.1), ('beta', 0.7000000000000001), ('gamma', 0.30000000000000004)]) 1.78
OrderedDict([('alpha', 0.1), ('beta', 0.7000000000000001), ('gamma', 0.5)]) 0.42
OrderedDict([('alpha', 0.1), ('beta', 0.7000000000000001), ('gamma', 0.7000000000000001)]) -0.34
OrderedDict([('alpha', 0.1), ('beta', 0.7000000000000001), ('gamma', 0.9)]) 0.56
OrderedDict([('alpha', 0.1), ('beta', 0.9), ('gamma', 0.1)]) 0.74
OrderedDict([('alpha', 0.1), ('beta', 0.9), ('gamma', 0.30000000000000004)]) 1.34
OrderedDict([('alpha', 0.1), ('beta', 0.9), ('gamma', 0.5)]) 0.86
OrderedDict([('alpha', 0.1), ('beta', 0.9), ('gamma', 0.7000000000000001)]) -0.94
OrderedDict([('alpha', 0.1), ('beta', 0.9), ('gamma', 0.9)]) 0.1
OrderedDict([('alpha', 0.30000000000000004), ('beta', 0.1), ('gamma', 0.1)]) 2.74
OrderedDict([('alpha', 0.30000000000000004), ('beta', 0.1), ('gamma', 0.30000000000000004)]) 2.94
OrderedDict([('alpha', 0.30000000000000004), ('beta', 0.1), ('gamma', 0.5)]) 0.5
OrderedDict([('alpha', 0.30000000000000004), ('beta', 0.1), ('gamma', 0.7000000000000001)]) -0.2
OrderedDict([('alpha', 0.30000000000000004), ('beta', 0.1), ('gamma', 0.9)]) -0.46
OrderedDict([('alpha', 0.30000000000000004), ('beta', 0.30000000000000004), ('gamma', 0.1)]) 6.34
OrderedDict([('alpha', 0.30000000000000004), ('beta', 0.30000000000000004), ('gamma', 0.30000000000000004)]) 5.82
OrderedDict([('alpha', 0.30000000000000004), ('beta', 0.30000000000000004), ('gamma', 0.5)]) 2.1
OrderedDict([('alpha', 0.30000000000000004), ('beta', 0.30000000000000004), ('gamma', 0.7000000000000001)]) 0.74
OrderedDict([('alpha', 0.30000000000000004), ('beta', 0.30000000000000004), ('gamma', 0.9)]) -0.36
OrderedDict([('alpha', 0.30000000000000004), ('beta', 0.5), ('gamma', 0.1)]) 7.8
OrderedDict([('alpha', 0.30000000000000004), ('beta', 0.5), ('gamma', 0.30000000000000004)]) 7.32
OrderedDict([('alpha', 0.30000000000000004), ('beta', 0.5), ('gamma', 0.5)]) 1.72
OrderedDict([('alpha', 0.30000000000000004), ('beta', 0.5), ('gamma', 0.7000000000000001)]) 0.6
OrderedDict([('alpha', 0.30000000000000004), ('beta', 0.5), ('gamma', 0.9)]) -0.42
OrderedDict([('alpha', 0.30000000000000004), ('beta', 0.7000000000000001), ('gamma', 0.1)]) 5.7
OrderedDict([('alpha', 0.30000000000000004), ('beta', 0.7000000000000001), ('gamma', 0.30000000000000004)]) 5.68
OrderedDict([('alpha', 0.30000000000000004), ('beta', 0.7000000000000001), ('gamma', 0.5)]) 2.06
OrderedDict([('alpha', 0.30000000000000004), ('beta', 0.7000000000000001), ('gamma', 0.7000000000000001)]) 0.14
OrderedDict([('alpha', 0.30000000000000004), ('beta', 0.7000000000000001), ('gamma', 0.9)]) -0.58
OrderedDict([('alpha', 0.30000000000000004), ('beta', 0.9), ('gamma', 0.1)]) 1.94
OrderedDict([('alpha', 0.30000000000000004), ('beta', 0.9), ('gamma', 0.30000000000000004)]) 2.84
OrderedDict([('alpha', 0.30000000000000004), ('beta', 0.9), ('gamma', 0.5)]) 1.64
OrderedDict([('alpha', 0.30000000000000004), ('beta', 0.9), ('gamma', 0.7000000000000001)]) -0.74
OrderedDict([('alpha', 0.30000000000000004), ('beta', 0.9), ('gamma', 0.9)]) 0.56
OrderedDict([('alpha', 0.5), ('beta', 0.1), ('gamma', 0.1)]) 2.38
OrderedDict([('alpha', 0.5), ('beta', 0.1), ('gamma', 0.30000000000000004)]) 3.2
OrderedDict([('alpha', 0.5), ('beta', 0.1), ('gamma', 0.5)]) 1.76
OrderedDict([('alpha', 0.5), ('beta', 0.1), ('gamma', 0.7000000000000001)]) -0.14
OrderedDict([('alpha', 0.5), ('beta', 0.1), ('gamma', 0.9)]) -0.34
OrderedDict([('alpha', 0.5), ('beta', 0.30000000000000004), ('gamma', 0.1)]) 8.18
OrderedDict([('alpha', 0.5), ('beta', 0.30000000000000004), ('gamma', 0.30000000000000004)]) 6.22
OrderedDict([('alpha', 0.5), ('beta', 0.30000000000000004), ('gamma', 0.5)]) 2.12
OrderedDict([('alpha', 0.5), ('beta', 0.30000000000000004), ('gamma', 0.7000000000000001)]) 0.48
OrderedDict([('alpha', 0.5), ('beta', 0.30000000000000004), ('gamma', 0.9)]) -0.08
OrderedDict([('alpha', 0.5), ('beta', 0.5), ('gamma', 0.1)]) 9.18
OrderedDict([('alpha', 0.5), ('beta', 0.5), ('gamma', 0.30000000000000004)]) 6.58
OrderedDict([('alpha', 0.5), ('beta', 0.5), ('gamma', 0.5)]) 1.98
OrderedDict([('alpha', 0.5), ('beta', 0.5), ('gamma', 0.7000000000000001)]) 0.08
OrderedDict([('alpha', 0.5), ('beta', 0.5), ('gamma', 0.9)]) -0.2
OrderedDict([('alpha', 0.5), ('beta', 0.7000000000000001), ('gamma', 0.1)]) 7.02
OrderedDict([('alpha', 0.5), ('beta', 0.7000000000000001), ('gamma', 0.30000000000000004)]) 4.56
OrderedDict([('alpha', 0.5), ('beta', 0.7000000000000001), ('gamma', 0.5)]) 0.32
OrderedDict([('alpha', 0.5), ('beta', 0.7000000000000001), ('gamma', 0.7000000000000001)]) -0.56
OrderedDict([('alpha', 0.5), ('beta', 0.7000000000000001), ('gamma', 0.9)]) -0.16
OrderedDict([('alpha', 0.5), ('beta', 0.9), ('gamma', 0.1)]) 2.76
OrderedDict([('alpha', 0.5), ('beta', 0.9), ('gamma', 0.30000000000000004)]) 1.74
OrderedDict([('alpha', 0.5), ('beta', 0.9), ('gamma', 0.5)]) 0.96
OrderedDict([('alpha', 0.5), ('beta', 0.9), ('gamma', 0.7000000000000001)]) 0.18
OrderedDict([('alpha', 0.5), ('beta', 0.9), ('gamma', 0.9)]) -0.44
OrderedDict([('alpha', 0.7000000000000001), ('beta', 0.1), ('gamma', 0.1)]) 1.82
OrderedDict([('alpha', 0.7000000000000001), ('beta', 0.1), ('gamma', 0.30000000000000004)]) 1.44
OrderedDict([('alpha', 0.7000000000000001), ('beta', 0.1), ('gamma', 0.5)]) 0.82
OrderedDict([('alpha', 0.7000000000000001), ('beta', 0.1), ('gamma', 0.7000000000000001)]) 0.14
OrderedDict([('alpha', 0.7000000000000001), ('beta', 0.1), ('gamma', 0.9)]) 0.24
OrderedDict([('alpha', 0.7000000000000001), ('beta', 0.30000000000000004), ('gamma', 0.1)]) 6.02
OrderedDict([('alpha', 0.7000000000000001), ('beta', 0.30000000000000004), ('gamma', 0.30000000000000004)]) 4.74
OrderedDict([('alpha', 0.7000000000000001), ('beta', 0.30000000000000004), ('gamma', 0.5)]) 2.44
OrderedDict([('alpha', 0.7000000000000001), ('beta', 0.30000000000000004), ('gamma', 0.7000000000000001)]) -0.06
OrderedDict([('alpha', 0.7000000000000001), ('beta', 0.30000000000000004), ('gamma', 0.9)]) -0.28
OrderedDict([('alpha', 0.7000000000000001), ('beta', 0.5), ('gamma', 0.1)]) 5.98
OrderedDict([('alpha', 0.7000000000000001), ('beta', 0.5), ('gamma', 0.30000000000000004)]) 3.32
OrderedDict([('alpha', 0.7000000000000001), ('beta', 0.5), ('gamma', 0.5)]) 0.84
OrderedDict([('alpha', 0.7000000000000001), ('beta', 0.5), ('gamma', 0.7000000000000001)]) -0.02
OrderedDict([('alpha', 0.7000000000000001), ('beta', 0.5), ('gamma', 0.9)]) 0.26
OrderedDict([('alpha', 0.7000000000000001), ('beta', 0.7000000000000001), ('gamma', 0.1)]) 4.74
OrderedDict([('alpha', 0.7000000000000001), ('beta', 0.7000000000000001), ('gamma', 0.30000000000000004)]) 1.7
OrderedDict([('alpha', 0.7000000000000001), ('beta', 0.7000000000000001), ('gamma', 0.5)]) 0.48
OrderedDict([('alpha', 0.7000000000000001), ('beta', 0.7000000000000001), ('gamma', 0.7000000000000001)]) -0.66
OrderedDict([('alpha', 0.7000000000000001), ('beta', 0.7000000000000001), ('gamma', 0.9)]) 0.08
OrderedDict([('alpha', 0.7000000000000001), ('beta', 0.9), ('gamma', 0.1)]) 1.92
OrderedDict([('alpha', 0.7000000000000001), ('beta', 0.9), ('gamma', 0.30000000000000004)]) 1.1
OrderedDict([('alpha', 0.7000000000000001), ('beta', 0.9), ('gamma', 0.5)]) 1.12
OrderedDict([('alpha', 0.7000000000000001), ('beta', 0.9), ('gamma', 0.7000000000000001)]) 0.22
OrderedDict([('alpha', 0.7000000000000001), ('beta', 0.9), ('gamma', 0.9)]) 0.38
OrderedDict([('alpha', 0.9), ('beta', 0.1), ('gamma', 0.1)]) 1.04
OrderedDict([('alpha', 0.9), ('beta', 0.1), ('gamma', 0.30000000000000004)]) 0.62
OrderedDict([('alpha', 0.9), ('beta', 0.1), ('gamma', 0.5)]) 0.12
OrderedDict([('alpha', 0.9), ('beta', 0.1), ('gamma', 0.7000000000000001)]) 0.02
OrderedDict([('alpha', 0.9), ('beta', 0.1), ('gamma', 0.9)]) 0.54
OrderedDict([('alpha', 0.9), ('beta', 0.30000000000000004), ('gamma', 0.1)]) 1.38
OrderedDict([('alpha', 0.9), ('beta', 0.30000000000000004), ('gamma', 0.30000000000000004)]) 0.08
OrderedDict([('alpha', 0.9), ('beta', 0.30000000000000004), ('gamma', 0.5)]) 0.66
OrderedDict([('alpha', 0.9), ('beta', 0.30000000000000004), ('gamma', 0.7000000000000001)]) -0.5
OrderedDict([('alpha', 0.9), ('beta', 0.30000000000000004), ('gamma', 0.9)]) -0.62
OrderedDict([('alpha', 0.9), ('beta', 0.5), ('gamma', 0.1)]) 1.76
OrderedDict([('alpha', 0.9), ('beta', 0.5), ('gamma', 0.30000000000000004)]) 1.62
OrderedDict([('alpha', 0.9), ('beta', 0.5), ('gamma', 0.5)]) -0.6
OrderedDict([('alpha', 0.9), ('beta', 0.5), ('gamma', 0.7000000000000001)]) -0.54
OrderedDict([('alpha', 0.9), ('beta', 0.5), ('gamma', 0.9)]) 0.42
OrderedDict([('alpha', 0.9), ('beta', 0.7000000000000001), ('gamma', 0.1)]) 1.1
OrderedDict([('alpha', 0.9), ('beta', 0.7000000000000001), ('gamma', 0.30000000000000004)]) 0.76
OrderedDict([('alpha', 0.9), ('beta', 0.7000000000000001), ('gamma', 0.5)]) -0.36
OrderedDict([('alpha', 0.9), ('beta', 0.7000000000000001), ('gamma', 0.7000000000000001)]) -0.96
OrderedDict([('alpha', 0.9), ('beta', 0.7000000000000001), ('gamma', 0.9)]) -0.56
OrderedDict([('alpha', 0.9), ('beta', 0.9), ('gamma', 0.1)]) 0.32
OrderedDict([('alpha', 0.9), ('beta', 0.9), ('gamma', 0.30000000000000004)]) 0.12
OrderedDict([('alpha', 0.9), ('beta', 0.9), ('gamma', 0.5)]) 1.1
OrderedDict([('alpha', 0.9), ('beta', 0.9), ('gamma', 0.7000000000000001)]) 0.34
OrderedDict([('alpha', 0.9), ('beta', 0.9), ('gamma', 0.9)]) 0.48
"""


# Finding the minimum 
# Now we have all the code, we do a simple grid search over values 
# to find a minimal value. Grid search is not the best optimization 
# algorithm, but is here simply illustrative.

sweep_size = 10
sweep = (cirq.Linspace(key='alpha', start=0.0, stop=1.0, length=sweep_size)
         * cirq.Linspace(key='beta', start=0.0, stop=1.0, length=sweep_size)
         * cirq.Linspace(key='gamma', start=0.0, stop=1.0, length=sweep_size))
results = simulator.run_sweep(circuit, params=sweep, repetitions=100)

min = None
min_params = None
for result in results:
    value = obj_func(result)
    if min is None or value < min:
        min = value
        min_params = result.params
print(f'Minimum objective value is {min}.')