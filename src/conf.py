""" This is a configuration file for the project that contains some basic 
settings like file paths to use for logging. """
import os 
LOG_FOLDER_PATH = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(LOG_FOLDER_PATH, exist_ok=True) 

# How many times should a given simulation be run? 
# NUM_SIMULATION_REPETITIONS simulations will run for each
# set of parameter values (alpha,beta,gamma). Those 
# NUM_SIMULATION_REPETITIONS simulations will provide a set of
# NUM_SIMULATION_REPETITIONS measurements which can be used to calculate
# the expectation value of the Hamiltonian based on the param values
# of (alpha,beta,gamma)
NUM_SIMULATION_REPETITIONS = 100