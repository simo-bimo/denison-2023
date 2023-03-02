import numpy as np
import numpy.ma as ma

from scipy import fft

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits import mplot3d

from typing import Callable
from typing import Tuple
from typing import List

class SoftSimulator:
    """
    An implementation of the Split Operator Fourier Transform Method to simulate
    a quantum harmonic oscillator.
    """

    # For the default QHO simulator:
    k = 1
    def QHO_potential(x: np.ndarray) -> np.ndarray:
        return 0.5 * SoftSimulator.k * (x**2)

    def __init__(self, 
                n_qubits=5, 
                length=10,
                potential_energy_function: Callable[[np.ndarray], np.ndarray] = QHO_potential,
                xoffset=1, 
                delta_t=1,
                steps_per_frame=1):
        """
        Generates a SoftSimulator object. Default potential energy function is for a quantum harmonic oscillator.
        """
        self.n_qubits = n_qubits
        self.n = 2**(self.n_qubits)
        self.length = length

        temp_grid = np.linspace(-length/2, length/2, self.n, endpoint=False)
        grid = np.meshgrid(temp_grid, temp_grid, temp_grid)
        self.discrete_grid = np.array(list(zip(*(x.flat for x in grid))))

        def mag(p: np.ndarray):
            return np.asarray(np.sqrt((p[:,0])**2 + p[:,1]**2 + p[:,2]**2))

        # print(f"Shape: {self.discrete_grid.shape}\n Shape of Radius: {mag(self.discrete_grid).shape}")

        self.delta_x = length / self.n

        # Physical constants
        self.hbar = 1.0
        self.m = 1.0

        # Number of momentum bases to consider.
        """
        The biggest momentum values that's considered should be inversely proportional to delta x,
        as a higher momentum corresponds to a higher frequency, and therefore requires a small delta x
        to be present in the position basis of the wavefunction.
        """
        self.kmax = np.pi / self.delta_x
        temp_grid = np.linspace(-self.kmax, self.kmax, self.n, endpoint=False)
        temp_grid = np.linspace(-length/2, length/2, self.n, endpoint=False)
        x, y, z = np.meshgrid(temp_grid, temp_grid, temp_grid)
        x = fft.ifftshift(x)
        y = fft.ifftshift(y)
        z = fft.ifftshift(z)
        self.momentum_grid = np.array(list(zip(*(x.flat for x in (x, y, z)))))

        # self.pos_psi = np.zeros(self.n, dtype=complex)

        # Radial magnitude but with x shifted for an initial psi.
        def mag_shift(p: np.ndarray):
            return np.asarray(np.sqrt((p[:,0]-xoffset)**2 + p[:,1]**2 + p[:,2]**2))

        self.pos_psi = self.normalise(np.exp(-(mag_shift(self.discrete_grid)**2)/2, dtype=complex))

        self.V_func = potential_energy_function
        
        self.V_operator = np.exp(-1j * potential_energy_function(mag(self.discrete_grid)) * delta_t / self.hbar)
        self.T_operator = np.exp(-1j * (mag(self.momentum_grid) ** 2) * delta_t / (2 * self.hbar * self.m))

        self.VE_operator = potential_energy_function(mag(self.discrete_grid))
        self.TE_operator = - self.momentum_grid**2 / (2 * self.hbar * self.m)

        self.steps_per_frame = steps_per_frame
        self.delta_t = delta_t

        pass
    
    def set_psi(self, psi: np.ndarray):
        self.pos_psi = self.normalise(psi)
        pass

    def position_to_momentum(self, position_psi: np.ndarray):
        return fft.fftn(position_psi, norm='ortho')

    def momentum_to_position(self, momentum_psi: np.ndarray):
        return fft.ifft(momentum_psi, norm='ortho')

    def next_time_step(self):
        """
        Evolves the system delta_t time forward using the split operator fourier transform method.
        """
        self.pos_psi = self.V_operator * self.pos_psi

        momentum_psi = self.position_to_momentum(self.pos_psi)

        momentum_psi = self.T_operator * momentum_psi

        self.pos_psi = self.momentum_to_position(momentum_psi)

        pass

    # Utility Plotting Functions
    def animate(self, 
                duration: int,
                cutoff=0.03):
        """
        Generates and returns a matplotlib animation of the current system up to 'duration' time steps.
        """
 
        fig = plt.figure()
        # axis = plt.axes(xlim=(-self.length/2, self.length/2), ylim=pos_ylim)
        axis = plt.axes(projection='3d')

        values = self.get_masked_position(cutoff=cutoff)
        x = ma.array(self.discrete_grid[:,0], mask=values.mask)
        y = ma.array(self.discrete_grid[:,1], mask=values.mask)
        z = ma.array(self.discrete_grid[:,2], mask=values.mask)

        graph = axis.scatter3D(x, y, z, c=values, cmap='hot')

        axis.set_xlim((-self.length/2, self.length/2))
        axis.set_ylim((-self.length/2, self.length/2))
        axis.set_zlim((-self.length/2, self.length/2))

        # Reset the mask after it's been plotted.
        # self.pos_psi.mask = False

        def animate_pos(num):
            values = self.get_masked_position(cutoff=cutoff)
            x = ma.array(self.discrete_grid[:,0], mask=values.mask)
            y = ma.array(self.discrete_grid[:,1], mask=values.mask)
            z = ma.array(self.discrete_grid[:,2], mask=values.mask)

            for i in range(self.steps_per_frame):
                self.next_time_step()
            
            xlim = axis.get_xlim3d()
            ylim = axis.get_ylim3d()
            zlim = axis.get_zlim3d()

            axis.clear()

            graph = axis.scatter3D(x, y, z, c=values, cmap='hot')

            axis.set_xlim(xlim)
            axis.set_ylim(ylim)
            axis.set_zlim(zlim)

            axis.set_xlabel('X')
            axis.set_ylabel('Y')
            axis.set_zlabel('Z')

            # self.pos_psi.mask = False

            return graph

        return FuncAnimation(fig, animate_pos, frames=int(duration/self.steps_per_frame), blit=False, repeat=False)

    def plot_energy(self, color='orange'):
        plt.plot(self.discrete_grid, self.V_func(self.discrete_grid), color=color)
        pass
    
    # Consider making these not a copy, for speed.
    def get_masked_position(self, cutoff=0.03):
        return ma.masked_where(np.abs(self.pos_psi) <= cutoff, self.pos_psi, copy = False)

    def get_masked_momentum(self, cutoff=0.03):
        return ma.masked_where(np.abs(self.position_to_momentum(self.pos_psi)) <= cutoff, self.position_to_momentum(self.pos_psi), copy = False)

    def plot_psi_position(self, color='blue', plot_func: Callable[[np.ndarray], np.ndarray] = np.real, cutoff=0.03):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_title('Position Basis')

        values = self.get_masked_position(cutoff=cutoff)
        x = ma.array(self.discrete_grid[:,0], mask=values.mask)
        y = ma.array(self.discrete_grid[:,1], mask=values.mask)
        z = ma.array(self.discrete_grid[:,2], mask=values.mask)

        # self.pos_psi.mask = False

        ax.scatter3D(x, y, z, c=values, cmap='hot')
        pass

    def plot_psi_momentum(self, color='blue', plot_func: Callable[[np.ndarray], np.ndarray] = np.real, cutoff=0.03):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_title('Momentum Basis')
        
        values = self.get_masked_momentum(cutoff=cutoff)
        x = ma.array(self.discrete_grid[:,0], mask=values.mask)
        y = ma.array(self.discrete_grid[:,1], mask=values.mask)
        z = ma.array(self.discrete_grid[:,2], mask=values.mask)

        ax.scatter3D(x, y, z, c=values, cmap='hot')
        pass

    def plot_psi_crosssection(self, cutoff = 0.05):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_title('Cross Section in XY plane of Position Basis')

        x = []
        y = []
        values = []
        for i in range(len(self.pos_psi)):
            if (np.abs(self.discrete_grid[i][2]) < cutoff ):
                x.append(self.discrete_grid[i][0])
                y.append(self.discrete_grid[i][1])
                values.append(self.pos_psi[i])

        ax.scatter3D(x, y, values, c=values, cmap='Greens')

        pass
    
    def normalise(self, psi: np.ndarray) -> np.ndarray:
        return psi / (np.sqrt(np.vdot(psi, psi) * self.delta_x**3))

    def get_energy(self) -> np.cdouble:
        epsi = self.VE_operator * self.pos_psi + self.TE_operator * self.pos_psi
        # We just grab the middle value.
        E = epsi[len(epsi)//2] / self.pos_psi[len(epsi)//2]
        return np.abs(E)

    # Useful Static Functions
    def probability_density(psi: np.ndarray) -> np.ndarray:
        return np.real(psi)**2 + np.imag(psi)**2

# Animator

# Simulation Parameters are a tuple of: (n_qubits, length, dx, vmax, delta_t, steps_per_frame).
SimParams = Tuple[int, np.double, np.double, np.double, np.double, int]

def vary_parameter_animation(params: SimParams,
                            param_index: int,
                            values: np.ndarray,
                            colors: List[str],
                            duration: int, 
                            plot_func: Callable[[np.ndarray], np.ndarray] = np.real, 
                            ylim=(0, 1),
                            xlim=(0, 1), 
                            plot_V = False):
    """
    Generates an animation of n coulomb simulations varying the 'param_index'th parameter of them across 'values'.
    """
    n = len(values)
    p_delta_t = params[4]
    p_steps_per_frame = params[5]

    if p_delta_t == 0:
        p_delta_t = 0.001

    if p_steps_per_frame == 0:
        p_steps_per_frame = 50

    params = list(params)
    params[4] = p_delta_t
    params[5] = p_steps_per_frame
    params = tuple(params)

    if param_index == 4:
        default_dt = min(values)
    else:
        default_dt = p_delta_t

    if not colors:
        colors = ['blue'] * n

    fig = plt.figure()
    if params[1] == 0:
        axis = plt.axes(xlim=xlim, ylim=ylim)
    else:
        axis = plt.axes(xlim=(-params[1]/2, params[1]/2), ylim=ylim)


    simulators = []

    for i in range(n):
        temp_param = list(params)
        temp_param[param_index] = values[i]
        temp_param = tuple(temp_param)
        simulators.append(generate_coulomb_simulator(temp_param))
        
        if plot_V:
            axis.plot(simulators[i].discrete_grid, simulators[i].V_func(simulators[i].discrete_grid), c=colors[i], linestyle='dashed', lw=1)


    lines = []
    for i in range(n):
            lobj, = axis.plot([], [], c=colors[i], lw=1, label=f'{values[i]}')
            lines.append(lobj)
    
    fig.legend()

    def animate_all(i):
        for i in range(n):
            lines[i].set_data(simulators[i].discrete_grid, plot_func(simulators[i].pos_psi))
            for j in range(int(simulators[i].steps_per_frame / simulators[i].delta_t * default_dt)):
                simulators[i].next_time_step()
        return lines

    return FuncAnimation(fig, animate_all, frames=int(duration/p_steps_per_frame), blit=False, repeat=False)

def generate_coulomb_simulator(params: SimParams, vmax_proportional_x=True):
    """
    Generates a soft simulator object with parameters ordered as follows:
    (n_qubits, length, dx, vmax)
    dx is only considered if length is 0.
    """

    # We need to pre-declare all these variables so we can use them
    # To define the coulomb potential that aligns with the simulator.
    n = params[0]
    length = params[1]
    p_vmax = params[3]
    p_delta_t = params[4]
    p_steps_per_frame = params[5]

    if length == 0:
        dx = params[2]
        length = dx * 2**n
    else:
        dx = length / 2**n

    x_grid = np.linspace(-length/2, length/2, 2**n, endpoint=False)

    def normalise(psi: np.ndarray) -> np.ndarray:
        return psi / (np.sqrt(np.vdot(psi, psi) * dx))

    if vmax_proportional_x:
        vmax = p_vmax*dx
    else:
        vmax = p_vmax

    def coulomb_potential(x: np.ndarray) -> np.ndarray:
        return -1 / (np.abs(x) + vmax)

    sim = SoftSimulator(n_qubits=n, potential_energy_function = coulomb_potential, length = length, delta_t = p_delta_t, steps_per_frame=p_steps_per_frame)
    sim.set_psi(np.exp(-(sim.discrete_grid-1.0) ** 2 / 2, dtype=complex))
    return sim

# Tests

def coulomb_test():
    def coulomb(x: np.ndarray) -> np.ndarray:
        return - 1 / (np.abs(x) + 0.0001)

    sim = SoftSimulator(n_qubits=5, length=40, delta_t = 0.005, steps_per_frame=1, xoffset=0, potential_energy_function=coulomb)
    anim = sim.animate(400, cutoff = 0.00)
    anim.save("coulomb test.gif", 'pillow', fps=30)
    pass

def QHO_test():
    sim = SoftSimulator(n_qubits=5, length=40, delta_t = 0.005, steps_per_frame=1, xoffset=0)
    anim = sim.animate(400, cutoff = 0.01)
    anim.save("QHO_test.gif", 'pillow', fps=30)
    pass

def main():
    print("Welcome to the simulator!")

    coulomb_test()
    # sim.plot_psi_momentum()

    plt.show()


    print("Simulation finished and saved.")
    pass
    
if __name__=="__main__":
    main()
