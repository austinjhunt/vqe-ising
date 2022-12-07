

""" 
solver.py 

This module demonstrates use of the Variational Quantum Eigensolver (VQE)
algorithm to find the ground state of a 2D +/- Ising model with a transverse field.
It uses Google's cirq library to create and simulate quantum circuit execution to handle
the quantum portion of the VQE algorithm. 

It is based on the Google Quantum AI documentation found here: 
https://quantumai.google/cirq/experiments/variational_algorithm

But this module rearchitects the code they provide in a more expressive, object-oriented form
primarily to assist with understanding the logical flow of the VQE algorithm. 

The Driver in main.py instantiates one of these solvers, passing in a grid size to use 
for the Ising model. Then the driver executes the solver.simulate() method to run the simulation
and identify the minimum energy level (ground state energy) of a given instance of the 
Ising model. The simulate method of the solver returns a dictionary containing that 
minimum energy as well as the parameters that resulted in that minimum energy. 
"""

import cirq
import random 
import numpy as np
from sympy import Symbol
from base import Base
from conf import NUM_SIMULATION_REPETITIONS

class VQEIsingSolver(Base):
    """ VQEIsingSolver is a class that encapsulates logic for using VQE 
    algorithm to solve an instance of the 2D +/- Ising problem. It inherits
    from Base for logging purposes. """

    def __init__(self, 
        name: str = 'VQEIsingSolver', # name for logging purposes
        verbose: bool = False, # whether to use verbose logging
        ising_grid_size: int = 3): # size of the square model grid
        super().__init__(name, verbose)
        # Init the ising model grid size
        self.ising_grid_size = ising_grid_size 
        
        self.h = None # 2D array (list of lists) representing transverse field terms
        self.jr = None # 2D array (list of lists) representing inter-row links
        self.jc = None # 2D array (list of lists) representing inter-column links

    def create_square_qubit_grid(self):
        """ Create a square lattice of qubits. The 2D Ising 
        problem has a natural grid structure. 
        Args: 
        
        Returns: 
        list - A list of GridQubits filling in a square grid. Each GridQubit
        in the list  will have a .row and .col property indicating their 
        position on the grid. 
        """
        return cirq.GridQubit.square(
            self.ising_grid_size)

    
    def build_random_problem_instance(self): 
        """ 
        We have a lot of freedom in how we define our variational ansatz. 
        In this case, we will use a variation of a Quantum Approximate 
        Optimization Algorithm (QAOA) technique to define an ansatz that is 
        related to our specific Ising problem. 

        We first need to identify how the problem instances will be represented 
        (i.e. what are the parameters that we need to encode and optimize to find the 
        ground state?). With our 2D +/- Ising model, these are the values J and h in the 
        Hamiltonian definition: 
        H = [ SUM_<i,j> { J_{i,j}*Z_i*Z_j } ] + [ SUM_i { h_i * Z_i } ]
        
        We represent both J and h as 2D arrays (lists of lists). So we
        use a function build_random_2d_array that generates a 
        random 2D array of dimension rows x cols. 

        Args: 


        Returns: 
        { 
            'transverse_field_terms': list of lists representing transverse field terms
            'links_in_row': list of lists representing inter-row links in Ising model
            'links_in_col': list of lists representing inter-column links in Ising model 
        }
        """

        def build_random_2d_array(rows, cols):
            """ Build a random 2D array (list of lists) comprising """
            return [[random.choice([+1, -1]) for _ in range(cols)] for _ in range(rows)]
         
        # build h, the transverse field terms
        h_transverse_field_terms = build_random_2d_array(
            rows=self.ising_grid_size, cols=self.ising_grid_size)
        # build j_r links within a row (there are rows - 1 inter-row links)
        jr_links_in_row = build_random_2d_array(
            rows=self.ising_grid_size- 1, cols=self.ising_grid_size)
        # links within a column ((there are col - 1 inter-col links))
        jc_links_in_col = build_random_2d_array(rows=self.ising_grid_size, cols=self.ising_grid_size - 1)
        return {
            'transverse_field_terms': h_transverse_field_terms,
            'links_in_row': jr_links_in_row,
            'links_in_col': jc_links_in_col
        }
 
    def initialize_ansatz(self, x_half_turns: Symbol = None, h_half_turns = None, j_half_turns: Symbol = None): 
        """ First step of VQE is creating a parameterized (via sympy) "ansatz" (a trial wave function) 
        that essentially encodes the problem in question into qubits.
        This ansatz is an educated guess about the parameters for our Parameterized 
        Quantum Circuit (PQC). 
        Args:
        x_half_turns - sympy.Symbol - parameterized (non-static) number of half turns to apply with X rotation gate in step 4. Value range driven by sympy.
        h_half_turns - sympy.Symbol - parameterized (non-static) number of half turns to apply with Z rotation gate in step 2. Value range driven by sympy. 
        j_half_turns - sympy.Symbol - parameterized (non-static) number of rotations about |11> conditioned on jr, jc to apply in step 3. Value range driven by sympy.
        
        Ansatz  will consist of two sub-circuits. 
        Sub-circuit 1 is is step one.  
        Sub-circuit 2 is steps 2-4
        
        Step 1. Apply an initial mixing step that puts all qubits into the 
        |+> = 1/sqrt(2) (|0>+|1>) state. (i.e., a superposition achieved with Hadamard gate)

        Step 2. Apply a cirq.ZPowGate for the same parameter for all qubits where 
        the transverse field term h is +1

        Step 3. Apply a cirq.CZPowGate for the same parameter between all qubits where the 
        coupling field term J is +1. If the field is -1, apply cirq.CZPowGate conjugated by X
        gates on all qubits.

        Step 4. Apply an cirq.XPowGate for the same parameter for all qubits.

        Returns: 
        cirq.Circuit - the ansatz / educated initial guess with initial parameter values 
        """

        def step_one():
            """ 
            Use a generator to yield a parameterized sub-circuit for a layer of the VQE algorithm. 
            The sub-circuit yielded (defined by the parameters) can subsequently be appended to a 
            cirq.Cirquit. 

            This function initializes superposition state with Hadamard gate on each grid qubit 
            of Ising model. 

            Args:
            
            Yields:
            parameterized sub-circuit of Hadamard gates for superposition state initialization
            """
            for i in range(self.ising_grid_size):
                for j in range(self.ising_grid_size):
                    yield cirq.H(cirq.GridQubit(i,j))

        def step_two(h: list = [], num_half_turns: float = 0.1):
            """
            Use a generator to yield a parameterized sub-circuit for a layer of the VQE algorithm. 
            The sub-circuit yielded (defined by the parameters) can subsequently be appended to a 
            cirq.Cirquit. 

            This function applies a cirq.ZPowGate for the same parameter for all qubits where 
            the transverse field term h is +1. 

            This function yields Z rotations by num_half_turns conditioned on the field h.
            (Applies a gate that rotates around the Z axis of the Bloch sphere)

            Args: 
            h: list - h, the transverse field terms 
            num_half_turns: float - number of half turns to apply with Z rotation gate.

            """
            gate = cirq.ZPowGate(exponent=num_half_turns)
            for i, h_row in enumerate(h):
                for j, h_ij in enumerate(h_row):
                    if h_ij == 1:
                        yield gate(cirq.GridQubit(i, j))

        def step_three(jr: list = [], jc: list = [], num_half_turns: float = 0.1):
            """
            Use a generator to yield a parameterized sub-circuit for a layer of the VQE algorithm. 
            The sub-circuit yielded (defined by the parameters) can subsequently be appended to a 
            cirq.Cirquit. 

            This function applies a cirq.CZPowGate for the same parameter between all qubits where the 
            coupling field term J is +1. If the field is -1, it applies a cirq.CZPowGate conjugated by X
            gates on all qubits.

            Yields rotations about |11> conditioned on the jr and jc fields.
            Args:
            jr: list - 2D array (list of lists) representing inter-row links 
            jc: list - 2D array (list of lists) representing inter-column links
            num_half_turns: float - number of half turns to apply with the ZPowGate. 

            """
            cz_gate = cirq.CZPowGate(exponent=num_half_turns)    
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

        def step_four(num_half_turns: float = 0.1):
            """ 
            Use a generator to yield a parameterized sub-circuit for a layer of the VQE algorithm. 
            The sub-circuit yielded (defined by the parameters) can subsequently be appended to a 
            cirq.Cirquit. 

            This function creates a gate that rotates around the X axis of 
            the Bloch sphere, where the rotation is determined by num_half_turns (default = 0.1). 
            This is because the rotation gate cirq.XPowGate is specified in half turns (ht). 
            For a rotation about X axis, the gate is: 
            cos(ht * pi)I + i*sin(ht*pi)X
            Apply that gate to each qubit in the Ising grid. 

            Args: 

            num_half_turns: float - number of half turns to apply. This is the t in gate**t. 
            and determines how much the eigenvalues of the gate are phased by.  Note that 
            if num_half_turns is 1, this applies the Pauli X gate. 

            Yields: 
            Parameterized sub-circuit of X rotation gates

            """ 
            rot = cirq.XPowGate(exponent=num_half_turns)
            for i in range(self.ising_grid_size):
                for j in range(self.ising_grid_size):
                    yield rot(cirq.GridQubit(i, j))

        # Putting all together, we can create a step that uses just three parameters. 
        # Below is the code, which uses the generator for each of the layers 
        # (note to advanced Python users: this code does not contain a bug in 
        # using yield due to the auto flattening of the OP_TREE concept. 
        # Typically, one would want to use yield from here, but this is not necessary):

        def step_one_wrapper():
            """ This function wraps the step one sub-circuit such that we can append
            the sub-circuit to a cirq.Circuit. 
            Args:

            Yields:
            sub-circuit handling step one of initializing ansatz. 
            """
            yield step_one()

        def step_two_three_four_wrapper(h: list = [], jr: list = [], jc: list = [], 
            x_half_turns: float = 0.1, h_half_turns: float = 0.1, j_half_turns: float = 0.1):
            """ This function wraps steps 2, 3, and 4 such that the result can be appended as 
            a single sub-circuit to the parent cirq.Circuit.
            
            Args: 
            h: list - 2D array (list of lists) representing transverse field terms in Ising model
            jr: list - 2D array (list of lists) representing inter-row links in Ising model
            jc: list - 2D array (list of lists) representing inter-column links in Ising model
             
            h_half_turns - float - number of half turns to apply with Z rotation gate in step 2
            j_half_turns - float - number of rotations about |11> conditioned on jr, jc to apply in step 3
            x_half_turns - float - number of half turns to apply with X rotation gate in step 4

            Yields:
            sub-circuit containing steps 2, 3, and 4 for initializing ansatz. 
            """
            yield step_two(h=h, num_half_turns=h_half_turns)
            yield step_three(jr=jr, jc=jc, num_half_turns=j_half_turns)
            yield step_four(num_half_turns=x_half_turns)

        # First build a random problem instance
        problem_instance = self.build_random_problem_instance()
        self.h = problem_instance['transverse_field_terms']
        self.jr = problem_instance['links_in_row']
        self.jc = problem_instance['links_in_col']
        
        # Initialize a circuit
        ansatz_circuit = cirq.Circuit()  

        # Append the step one sub-circuit
        ansatz_circuit.append(step_one_wrapper())
        # Append the steps 2,3,4 sub-circuit
        ansatz_circuit.append(step_two_three_four_wrapper(
            h=self.h, jr=self.jr, jc=self.jc, 
            x_half_turns=x_half_turns, 
            h_half_turns=h_half_turns, 
            j_half_turns=j_half_turns)) 
        # Here we see that we have chosen particular parameter values 0.1, 0.2, 0.3 
        self.info(ansatz_circuit)
        return ansatz_circuit

    def energy_func(self, h: list = [], jr: list = [], jc: list = []):
        """ 
        This function calculates the objective (it wraps another function energy
        which takes in the measurement results). We can pass this energy_func to the 
        results.histogram method (where results comes from simulator.run)
        so that the Counter (a dictionary, essentially) returned from results.histogram 
        uses the measured energies as the keys. We can then use those values to calculate
        the expectation value (average of the measured energies) of the Hamiltonian.
        Args: 
        h: 2D array (list of lists) representing transverse field terms in Ising model
        jr: 2D array (list of lists) representing transverse field terms
        jc: 2D array (list of lists) representing inter-column links in Ising model

        Returns: 
        energy: function - function that calculates Hamiltonian
        """
        grid_length = self.ising_grid_size
        def energy(measurements: list = []):
            """ Get the energy (i.e. calculate the Hamiltonian) from measurements list.
            Args:

            measurements: list - list of measurements from simulation run

            Returns:
            total energy, or Hamiltonian 
            """
            # Reshape the measurements into an array that matches Ising grid shape.
            meas_list_of_lists = [measurements[i * grid_length:(i + 1) * grid_length]
                                for i in range(grid_length)]
            # Convert true/false to +1/-1.
            pm_meas = 1 - 2 * np.array(meas_list_of_lists).astype(np.int32)

            # Initialize total energy as the sum of the plus/minus measurements *
            # transverse field terms h. This comes from the term SUM_i (Z_i * h_i) in the 
            # Hamiltonian definition of Ising model. 
            tot_energy = np.sum(pm_meas * h) 
            
            # Now also add in the other term of the Hamiltonian SUM_<i,j> J_{i,j}*Z_i*Z_j.
            for i, jr_row in enumerate(jr):
                for j, jr_ij in enumerate(jr_row):
                    tot_energy += jr_ij * pm_meas[i, j] * pm_meas[i + 1, j]
            for i, jc_row in enumerate(jc):
                for j, jc_ij in enumerate(jc_row):
                    tot_energy += jc_ij * pm_meas[i, j] * pm_meas[i, j + 1]

            # That gives us the total energy, which we can return. 
            return tot_energy
        return energy

    def calculate_expectation_value(self, result: cirq.Result = None):
        """ Calculate the expectation value of the Hamiltonian from the results of 
        the measurements, where the expectation value is the average of all
        possible measurement outcomes for an observable (the Hamiltonian) weighted 
        by their respective probabilities. 
        
        We first obtain the histogram of the measured energies, where the resulting 
        Counter uses the energy as the key and the probability as the value. 

        We then take SUM_i [energy_i * energyprobability_i] and divide that 
        sum by the total number of simulation repetitions to get the average, i.e., 
        the expectation value. 
        
        Args:

        result: cirq.Result - result containing all the measurements for a given 
        simulation run. 

        Returns:
        expectation value - float - expectation value of the Hamiltonian with 
        these measurements
        """
        energies_histogram = result.histogram(
            key='x', 
            fold_func=self.energy_func(
                h=self.h,
                jr=self.jr,
                jc=self.jc 
            )
        )
        return np.sum([k * v for k,v in energies_histogram.items()]) / result.repetitions

    def simulate(self):
        """ 
        Run optimization to find minimum objective value of Hamiltonian (ground state energy of 
        the Ising model). This can be done by parameterizing the ansatz circuit. 
        
        On quantum hardware, one would most likely want to have the optimization code 
        as close to the hardware as possible. As the classical hardware that is 
        allowed to inter-operate with the quantum hardware becomes better specified, 
        this language will be better defined. Without this specification, however, 
        Cirq also provides a useful concept for optimizing the looping in many 
        optimization algorithms. This is the fact that many of the values in the gate sets can, instead of being specified by a float, be specified by a symply.Symbol, and this sympy.Symbol 
        can be substituted for a value specified at execution time! 

        In essence, paramaterizing our values for our Parameterized Quantum Circuit (PQC) can 
        be handled by passing sympy.Symbol objects rather than passing static float values. 
        """
        # Create a simulator with cirq
        simulator = cirq.Simulator()

        # Create a square grid / lattice of qubits (into which we can encode
        # our ising problem in order to use VQE)
        qubits_grid = self.create_square_qubit_grid()
        
        # These three variables represent gate rotation params 
        # which are the parameters we are tuning for the optimization. 
        alpha, beta, gamma = (Symbol(_) for _ in ['alpha', 'beta', 'gamma'])
        # Parameterize the circuit gates with sympy Symbols as the half turns
        # The parameter vals are specified at runtime using a cirq.ParamResolver,
        # which is just a dictionary from Symbol keys to runtime values.
        ansatz_circuit = self.initialize_ansatz(
            x_half_turns=alpha, 
            h_half_turns=beta, 
            j_half_turns=gamma
        )
        ansatz_circuit.append(cirq.measure(*qubits_grid, key='x'))

        
        # Cirq also has the concept of a sweep. A sweep is a collection of 
        # parameter resolvers. This runtime information is very useful when 
        # one wants to run many circuits for many different parameter values.
        # Sweeps can be created to specify values directly (this is one way 
        # to get classical information into a circuit), or a variety of helper 
        # methods. For example suppose we want to evaluate our circuit 
        # over an equally spaced grid of parameter values (Grid Search used in ML)
        # We can easily create this using cirq.LinSpace. 

        # Finding the minimum 
        # Now we have all the code, we do a simple grid search over values 
        # to find a minimal value. Grid search is not the best optimization 
        # algorithm, but is here simply illustrative.

        sweep_size = 10
        sweep = (cirq.Linspace(key='alpha', start=0.0, stop=1.0, length=sweep_size)
                * cirq.Linspace(key='beta', start=0.0, stop=1.0, length=sweep_size)
                * cirq.Linspace(key='gamma', start=0.0, stop=1.0, length=sweep_size))
        results = simulator.run_sweep(
            ansatz_circuit, 
            params=sweep, 
            repetitions=NUM_SIMULATION_REPETITIONS
            )

        min = None
        min_params = None
        for result in results:
            value = self.calculate_expectation_value(result)
            if min is None or value < min:
                min = value
                min_params = result.params
        self.info(f'Minimum objective value is {min}. Params to produce minimum are: {min_params}')
        return {
            'min_energy': min, 
            'min_params': min_params
        }

if __name__ == "__main__":
    solver = VQEIsingSolver(ising_grid_size=3)
     
  


