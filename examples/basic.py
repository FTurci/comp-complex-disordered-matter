"""
Basic example showing how to run a simple Ising model simulation.
"""
from compdismatter.core import IsingModel

# Create and run a basic simulation
model = IsingModel(N=200, equilibration=1024, production=1024)

# Run the simulation
model.simulate(temperature=4.0)
model.plot_config()
