import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt

class IsingModel:
    def __init__(self, N, nt=88, eqSteps=1024, mcSteps=1024, T_range=(1.53, 3.28)):
        """
        Initialize the Ising model simulation.
        
        Parameters:
        -----------
        N : int
            Size of the lattice (N x N)
        nt : int
            Number of temperature points
        eqSteps : int
            Number of Monte Carlo sweeps for equilibration
        mcSteps : int
            Number of Monte Carlo sweeps for calculation
        T_range : tuple
            Temperature range (start, end) for simulation


        Example usage:
        
        # Initialize model
        model = IsingModel(N=16, nt=88, eqSteps=1024, mcSteps=1024)
        
        # Run the simulation
        model.simulate()
        
        # Plot the results
        model.plot_results()    
        """
        self.N = N
        self.nt = nt
        self.eqSteps = eqSteps
        self.mcSteps = mcSteps
        
        # Create temperature array
        self.T = np.linspace(T_range[0], T_range[1], nt)
        
        # Initialize arrays for observables
        self.E = np.zeros(nt)
        self.M = np.zeros(nt)
        self.C = np.zeros(nt)
        self.X = np.zeros(nt)
        
        # Normalization factors
        self.n1 = 1.0/(mcSteps*N*N)
        self.n2 = 1.0/(mcSteps*mcSteps*N*N)
    
    def initialstate(self):
        """ Generate a random spin configuration for initial condition """
        state = 2*np.random.randint(2, size=(self.N, self.N))-1
        return state
    
    def mcmove(self, config, beta):
        """ Monte Carlo move using Metropolis algorithm """
        N = self.N
        for i in range(N):
            for j in range(N):
                a = np.random.randint(0, N)
                b = np.random.randint(0, N)
                s = config[a, b]
                nb = config[(a+1)%N,b] + config[a,(b+1)%N] + config[(a-1)%N,b] + config[a,(b-1)%N]
                cost = 2*s*nb
                if cost < 0:
                    s *= -1
                elif rand() < np.exp(-cost*beta):
                    s *= -1
                config[a, b] = s
        return config
    
    def calcEnergy(self, config):
        """ Energy of a given configuration """
        energy = 0
        N = self.N
        for i in range(N):
            for j in range(N):
                S = config[i,j]
                nb = config[(i+1)%N, j] + config[i,(j+1)%N] + config[(i-1)%N, j] + config[i,(j-1)%N]
                energy += -nb*S
        return energy/4.
    
    def calcMag(self, config):
        """ Magnetization of a given configuration """
        return np.sum(config)
    
    def simulate(self):
        """ Run the simulation for all temperature points """
        for tt in range(self.nt):
            E1 = M1 = E2 = M2 = 0
            config = self.initialstate()
            iT = 1.0/self.T[tt]
            iT2 = iT*iT
            
            # Equilibration phase
            for i in range(self.eqSteps):
                self.mcmove(config, iT)
            
            # Measurement phase
            for i in range(self.mcSteps):
                self.mcmove(config, iT)
                Ene = self.calcEnergy(config)
                Mag = self.calcMag(config)
                
                E1 += Ene
                M1 += Mag
                M2 += Mag*Mag
                E2 += Ene*Ene
            
            # Calculate observables
            self.E[tt] = self.n1 * E1
            self.M[tt] = self.n1 * M1
            self.C[tt] = (self.n1 * E2 - self.n2 * E1 * E1) * iT2
            self.X[tt] = (self.n1 * M2 - self.n2 * M1 * M1) * iT
    
    def plot_results(self):
        """ Plot the results of the simulation """
        f = plt.figure(figsize=(18, 10))
        
        # Plot energy
        sp = f.add_subplot(2, 2, 1)
        plt.plot(self.T, self.E, 'o', color='blue')
        plt.xlabel("Temperature (T)", fontsize=20)
        plt.ylabel("Energy", fontsize=20)
        plt.axis('tight')
        
        # Plot magnetization
        sp = f.add_subplot(2, 2, 2)
        plt.plot(self.T, abs(self.M), 'o', color='red')
        plt.xlabel("Temperature (T)", fontsize=20)
        plt.ylabel("Magnetization", fontsize=20)
        plt.axis('tight')
        
        # Plot specific heat
        sp = f.add_subplot(2, 2, 3)
        plt.plot(self.T, self.C, 'o', color='green')
        plt.xlabel("Temperature (T)", fontsize=20)
        plt.ylabel("Specific Heat", fontsize=20)
        plt.axis('tight')
        
        # Plot susceptibility
        sp = f.add_subplot(2, 2, 4)
        plt.plot(self.T, self.X, 'o', color='purple')
        plt.xlabel("Temperature (T)", fontsize=20)
        plt.ylabel("Susceptibility", fontsize=20)
        plt.axis('tight')
        
        plt.tight_layout()
        plt.show()


