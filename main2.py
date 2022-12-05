"""_summary_
https://qiskit.org/textbook/ch-applications/vqe-molecules.html
"""

# pylint: disable=missing-function-docstring
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
import numpy as np
np.random.seed(999999)
p0 = np.random.random()
target_distr = {0: p0, 1: 1-p0}


def get_var_form(params):
    """ function that takes the parameters of our
    single U3 variational form as arguments and
    returns the corresponding quantum circuit"""
    qr = QuantumRegister(1, name="q")
    cr = ClassicalRegister(1, name='c')
    qc = QuantumCircuit(qr, cr)
    qc.u(params[0], params[1], params[2], qr[0])
    qc.measure(qr, cr[0])
    return qc


from qiskit import Aer
backend = Aer.get_backend("aer_simulator")
# Now we specify the objective function which takes as
# input a list of the variational form's parameters, and
# returns the cost associated with those parameters:
def counts_to_distr(counts):
    """Convert Qiskit result counts to dict with integers as
    keys, and pseudo-probabilities as values."""
    n_shots = sum(counts.values())
    return {int(k, 2): v/n_shots for k, v in counts.items()}

def objective_function(params):
    """Compares the output distribution of our circuit with
    parameters `params` to the target distribution."""
    # Create circuit instance with paramters and simulate it
    qc = get_var_form(params)
    result = backend.run(qc).result()
    # Get the counts for each measured state, and convert
    # those counts into a probability dict
    output_distr = counts_to_distr(result.get_counts())
    # Calculate the cost as the distance between the output
    # distribution and the target distribution
    cost = sum(
        abs(target_distr.get(i, 0) - output_distr.get(i, 0))
        for i in range(2**qc.num_qubits)
    )
    return cost


# Finally, we create an instance of the COBYLA optimizer,
# and run the algorithm. Note that the output varies from
# run to run due to random noise. Moreover, while close,
# the obtained distribution will not be exactly the same
# as the target distribution.

from qiskit.algorithms.optimizers import COBYLA
optimizer = COBYLA(maxiter=500, tol=0.0001)

# Create the initial parameters (noting that our
# single qubit variational form has 3 parameters)
params = np.random.rand(3)
result = optimizer.minimize(
    fun=objective_function,
    x0=params)

# Obtain the output distribution using the final parameters
qc = get_var_form(result.x)
counts = backend.run(qc, shots=10000).result().get_counts()
output_distr = counts_to_distr(counts)

print("Parameters Found:", result.x)
print("Target Distribution:", target_distr)
print("Obtained Distribution:", output_distr)
print("Cost:", objective_function(result.x))
