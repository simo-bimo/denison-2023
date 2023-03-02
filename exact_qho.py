from typing import Callable

import numpy as np

from scipy import special
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class ExactQHO:
    def __init__(self, n_qubits=9, length=10.0, k=1.0, m=1.0):
        self.n = 2**n_qubits

        # System constants
        self.k=k
        self.m=m
        self.hbar=1
        self.omega = np.sqrt(k / m)

        # number of fock states to simulate
        self.N = 50

        # includes both endpoints? either zero or both endpoints.
        self.discrete_grid = np.linspace(-length/2, length/2, self.n, endpoint=False)
        self.delta_x = np.abs(self.discrete_grid[1]-self.discrete_grid[0])
        self.psi_naught = np.zeros(self.n, dtype=complex)

        # The current time_step
        self.t = 0.0
        self.curr_psi = self.psi_naught
        self.curr_psi_fock = np.zeros(self.N, dtype=complex)

        pass

    def next_time_step(self, delta_t) -> np.ndarray:
        """
        Advances the wavefunction by delta_t time, returning the new wave function in a position basis.
        """
        next_psi = np.zeros(self.n, dtype=complex)
        for i in range(self.N):
            E = (i + 0.5) * self.omega * self.hbar
            next_psi += self.curr_psi_fock[i] * np.exp(-1j * E * delta_t / self.hbar) * self.nth_fock_basis(i)
        
        next_psi = self.normalise(next_psi)
        self.curr_psi = next_psi
        self.curr_psi_fock = self.pos_to_fock(self.curr_psi)

        self.t+=delta_t

        return self.curr_psi


    def plot(self, alpha=1):
        # plt.plot(self.discrete_grid, np.imag(self.curr_psi), color='red', alpha = alpha)
        plt.plot(self.discrete_grid, np.real(self.curr_psi), color='blue', alpha = alpha)
        pass
    
    def set_psi_naught_pos(self, psi: np.ndarray):
        """
        Sets Psi(x,0) of the system, assuming the provided psi is in the position basis.
        """
        if (len(psi) != self.n):
            raise ValueError("Incorrect length of initial position based psi")
        
        self.psi_naught = self.normalise(psi)

        self.curr_psi = self.psi_naught
        self.curr_psi_fock = self.pos_to_fock(self.curr_psi)
        pass
    
    def set_psi_naught_fock(self, psi: np.ndarray):
        """
        Sets Psi(x,0) of the system, assuming the provided psi is in the fock basis.
        """
        if (len(psi) > self.N):
            raise ValueError("Incorrect length of initial fock based psi.")
        pad_psi = np.pad(psi, (0, self.N-len(psi)), 'constant')

        self.curr_psi_fock = self.normalise(pad_psi)

        self.psi_naught = self.fock_to_pos(self.curr_psi_fock)

        self.curr_psi = self.psi_naught
        pass
    
    def pos_to_fock(self, position_psi: np.ndarray) -> np.ndarray:
        """
        Moves psi from position to fock basis.
        """
        new_psi = np.zeros(self.N, dtype=complex)

        for i in range(self.N):
            new_psi[i] = ExactQHO.function_dot(self.nth_fock_basis(i), position_psi) * self.delta_x

        return new_psi
    
    def fock_to_pos(self, fock_psi: np.ndarray) -> np.ndarray:
        """
        Moves psi from fock to position basis, assumes fock_psi is self.N long.
        """
        new_psi = np.zeros(self.n, dtype=complex)

        for i in range(self.N):
            new_psi += fock_psi[i] * self.nth_fock_basis(i)
        
        # We normalise again here because we lose detail by limiting the number of Fock States
        return self.normalise(new_psi)

    def nth_fock_basis(self, n: np.intc) -> np.ndarray:
        """
        Returns a psi(x), which is the normalised n'th eigenstate for a quantum harmonic oscillator.
        """
        alpha = self.m * self.omega / self.hbar

        herm = special.hermite(n)
        n = float(n)
        normaliser = np.power( (alpha / np.pi), 0.25)
        if (n > 0):
            normaliser /= ( np.sqrt((2**n) * np.math.factorial(n)) )

        x = self.discrete_grid
        return normaliser * herm(np.sqrt(alpha) * x) * np.exp(- alpha * (x**2) / 2)

    def normalise(self, psi: np.ndarray) -> np.ndarray:
        return psi / np.sqrt(np.vdot(psi, psi) * self.delta_x)

    # STATIC FUNCS
    def function_dot(basis_psi: np.ndarray, projected_psi: np.ndarray) -> np.cdouble:
        """
        Returns the normalised vdot product of a basis and projected_psi.
        Used to calculate coefficients of a superposition of eigenstates.
        """
        return np.vdot(basis_psi, projected_psi)# / np.vdot(basis_psi, basis_psi)
    
    
    def density(psi: np.ndarray) -> np.double:
        return np.real(psi)**2 + np.imag(psi)**2

# Manual Testing

TOTAL_STEPS = 200
DELTA_TIME = 0.01

def main():
    length = 8
    system = ExactQHO(9, length, k=3)
    system.set_psi_naught_pos(np.exp(-(system.discrete_grid-2.0)**2))
    
    # plt.plot(system.discrete_grid, ExactQHO.density(system.nth_fock_basis(15)))
    # plt.show()
    
    

    # for i in range(TOTAL_STEPS):
    #     system.next_time_step(DELTA_TIME)
    #     system.plot(alpha=i/TOTAL_STEPS)
    # plt.show()

    # Animations
    fig = plt.figure()

    axis = plt.axes(xlim=(-length/2, length/2), ylim=(0, 1.0))

    real, = axis.plot([], [], c='blue', lw=1)
    
    def animate(i):
        system.next_time_step(DELTA_TIME*i)
        real.set_data(system.discrete_grid, ExactQHO.density(system.curr_psi))

        return real,
    
    anim = FuncAnimation(fig, animate, frames=TOTAL_STEPS, blit=True)
    anim.save("A Wave Evolution.gif", 'pillow')

if __name__ == "__main__":
    main()
