""" 
main.py 

This file contains a Driver that is used to actually drive the 
instantiation of VQEIsingSolvers and the execution of their .simulate() methods. 
"""
from base import Base
from vqe_ising_solver.solver import VQEIsingSolver

class Driver(Base):
    def __init__(self, name: str = 'Driver', verbose: bool = False):
        super().__init__(name, verbose)
    
    def run_ising_solver(self, ising_grid_size: int = 3):
        """ Run an Ising solver that uses VQE to solve Ising problem 
        (i.e. finds ground state energy of Ising model that is 
        ising_grid_size x ising_grid_size) in dimension """
        self.info(
            f'Creating VQE Ising Solver for Ising model with '
            f'square grid size {ising_grid_size}x{ising_grid_size}')
        solver = VQEIsingSolver(name=f'VQEIsingSolver-{ising_grid_size}x{ising_grid_size}')
        result = solver.simulate()
        self.info(
            f'With a grid size of {ising_grid_size}, the minimum '
            f'energy level is {result["min_energy"]} corresponding '
            f'to the parameters {result["min_params"]}')
        return result
    
if __name__ == "__main__":
    driver = Driver()
    for grid_size in range(3,5): 
        result = driver.run_ising_solver(
            ising_grid_size=grid_size)
         