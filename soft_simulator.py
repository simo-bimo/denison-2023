import numpy as np

from scipy import fft

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
                n_qubits=10, 
                length=10,
                potential_energy_function: Callable[[np.ndarray], np.ndarray] = QHO_potential, 
                delta_t=1, 
                steps_per_frame=1):
        """
        Generates a SoftSimulator object. Default potential energy function is for a quantum harmonic oscillator.
        """
        self.n_qubits = n_qubits
        self.n = 2**self.n_qubits
        self.length = length
        # endpoint=False means we get a value at 0, since n is necessarily even.
        self.discrete_grid = np.linspace(-length/2, length/2, self.n, endpoint=False)
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
        self.momentum_grid = np.linspace(-self.kmax, self.kmax, self.n, endpoint=False)
        self.momentum_grid = fft.ifftshift(self.momentum_grid)

        # Default to a Gaussian that is translated 2 to the right.
        self.pos_psi = np.zeros(self.n, dtype=complex)
        self.pos_psi = self.normalise(np.exp((-(self.discrete_grid-2.0)**2)/2, dtype=complex))

        self.V_func = potential_energy_function
        
        self.V_operator = np.exp(-1j * potential_energy_function(self.discrete_grid) * delta_t / self.hbar)
        self.T_operator = np.exp(-1j * (self.momentum_grid ** 2) * delta_t / (2 * self.hbar * self.m))

        self.VE_operator = potential_energy_function(self.discrete_grid)
        self.TE_operator = - self.momentum_grid**2 / (2 * self.hbar * self.m)

        self.steps_per_frame = steps_per_frame
        self.delta_t = delta_t
        pass
    
    def set_psi(self, psi: np.ndarray):
        self.pos_psi = self.normalise(psi)
        pass

    def position_to_momentum(self, position_psi: np.ndarray):
        return fft.fft(position_psi, norm='ortho')

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
                plot_func: Callable[[np.ndarray], np.ndarray] = np.real, 
                pos_ylim=(0, 1), 
                plot_V = False,
                text_E = False):
        """
        Generates and returns a matplotlib animation of the current system up to 'duration' time steps.
        """
 
        fig = plt.figure()
        axis = plt.axes(xlim=(-self.length/2, self.length/2), ylim=pos_ylim)
            
        pos_line, = axis.plot([], [], c='blue', lw=1)

        if plot_V:
            axis.plot(self.discrete_grid, self.V_func(self.discrete_grid), c='orange', lw=1)

        def animate_pos(i):
            if text_E:
                for i in axis.texts:
                    i.remove()
                axis.text(1, 1, f'E: {self.get_energy():.2f}', fontsize=16)
            pos_line.set_data(self.discrete_grid, plot_func(self.pos_psi))
            for i in range(self.steps_per_frame):
                self.next_time_step()
            return pos_line,

        return FuncAnimation(fig, animate_pos, frames=int(duration/self.steps_per_frame), blit=True, repeat=False)

    def animate_momentum(self, 
                        duration: int, 
                        plot_func: Callable[[np.ndarray], np.ndarray] = np.real, 
                        pos_ylim=(0,1), 
                        mom_ylim=(0,1), 
                        plot_V = False):

        fig, axes = plt.subplots(2,1)
        
        axes[0].set_xlim(-self.length/2, self.length/2)
        axes[0].set_ylim(pos_ylim[0], pos_ylim[1])
        axes[0].set_title('Position Space')

        axes[1].set_xlim(-self.kmax, self.kmax)
        axes[1].set_ylim(mom_ylim[0], mom_ylim[1])
        axes[1].set_title('Momentum Space')

        if plot_V:
            axes[0].plot(self.discrete_grid, self.V_func(self.discrete_grid), c='orange', lw=1)
        
        position_line, = axes[0].plot([], [], c='blue', lw=1)
        momentum_line, = axes[1].plot([], [], c='blue', lw=1)

        def init():
            position_line.set_data(self.discrete_grid, plot_func(self.pos_psi))
            momentum_line.set_data(self.momentum_grid, plot_func(self.position_to_momentum(self.pos_psi)))
            return position_line, momentum_line

        def animate_pos_mom(i):
            for i in range(self.steps_per_frame):
                self.next_time_step()
            position_line.set_data(self.discrete_grid, plot_func(self.pos_psi))
            momentum_line.set_data(self.momentum_grid, plot_func(self.position_to_momentum(self.pos_psi)))
            return position_line, momentum_line
        
        return FuncAnimation(fig, animate_pos_mom, frames=int(duration/self.steps_per_frame), blit=True, repeat=False, init_func=init)

    def plot_energy(self, color='orange'):
        plt.plot(self.discrete_grid, self.V_func(self.discrete_grid), color=color)
        pass
    
    def plot_psi_position(self, color='blue', plot_func: Callable[[np.ndarray], np.ndarray] = np.real, plotter=plt):
        plotter.plot(self.discrete_grid, plot_func(self.pos_psi), color=color)
        # plotter.set(xlabel='Position')
        pass

    def plot_psi_momentum(self, color='red', plot_func: Callable[[np.ndarray], np.ndarray] = np.real, plotter=plt):
        # Reshift order to be plottable
        yvalues = (self.position_to_momentum(self.pos_psi))

        plotter.plot(self.momentum_grid, plot_func(yvalues), color=color)
        plotter.set(xlabel='Momentum')
        pass
    
    def normalise(self, psi: np.ndarray) -> np.ndarray:
        return psi / (np.sqrt(np.vdot(psi, psi) * self.delta_x))

    def get_energy(self) -> np.cdouble:
        # This feels like it's in the wrong basis or something.
        expected_V = np.vdot(self.pos_psi, self.VE_operator * self.pos_psi) * self.delta_x

        # H_x = self.VE_operator * self.pos_psi + self.TE_operator * self.position_to_momentum(self.pos_psi)

        # expected_H = np.vdot(self.pos_psi, H_x) * self.delta_x

        # return expected_H

        # V = a*r^n, so T = 0.5 * n * V
        # Does this really hold if we have the vmax cutoff?
        # In that case potential energy is quite of the right form.
        if self.V_func == SoftSimulator.QHO_potential:
            expected_T = expected_V
        else:
            # Assuming a coulomb potential
            expected_T = -0.5 * expected_V
        return expected_V + expected_T

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
    # sim = SoftSimulator(n_qubits=n, length = length, delta_t = p_delta_t, steps_per_frame=p_steps_per_frame)
    sim.set_psi(np.exp(-(sim.discrete_grid-1.0) ** 2 / 2, dtype=complex))
    return sim

# Tests

def test_fourier():
    sim = SoftSimulator(n_qubits=12, length = 10, delta_t=0.5, momentum_len=0.1)

    fig, axes = plt.subplots(2,1)

    sim.set_psi(np.exp(-(sim.discrete_grid-2)**2))
    sim.plot_psi_position(plot_func = SoftSimulator.probability_density, plotter=axes[0])
    sim.plot_psi_momentum(plot_func = SoftSimulator.probability_density, plotter=axes[1])
    plt.show()

    pass

def centered_gaussian_test():
    print("Rendering Centered Gaussian Test")

    sim = SoftSimulator(delta_t=1, steps_per_frame=10)

    sim.set_psi(np.exp((-(sim.discrete_grid)**2)/2))

    anim = sim.animate_momentum(1500, plot_func=np.real, plot_V=True, pos_ylim=(-3, 3))
    anim.save("SOFT Centered Gaussian Test.gif", 'pillow')
    # anim = sim.animate_momentum(1500, plot_func=SoftSimulator.probability_density, plot_V=True, pos_ylim=(0, 5))
    # anim.save("SOFT Centered Gaussian Test Density.gif", 'pillow')

    print("Test Complete")

    pass

def offset_gaussian_test():
    print("Rendering Offset Gaussian Test")

    sim = SoftSimulator(delta_t=0.01, steps_per_frame=2, length=10)

    sim.set_psi(np.exp((-(sim.discrete_grid-2.0)**2)/2, dtype=complex))

    anim = sim.animate_momentum(400, plot_func=SoftSimulator.probability_density, plot_V=True, pos_ylim=(0, 1))
    anim.save("SOFT Offset Gaussian Test.gif", 'pillow', fps=30)

    print("Test Complete")

    pass

def fat_offset_gaussian_test():
    print("Rendering fat offset gaussian test...")

    sim = SoftSimulator(delta_t=0.01, steps_per_frame=2, length=10)

    sim.set_psi(np.exp((-(sim.discrete_grid-2.0)**2)/3, dtype=complex))

    anim = sim.animate_momentum(400, plot_func=SoftSimulator.probability_density, plot_V=True, pos_ylim=(0, 1))
    anim.save("SOFT Fat Offset Gaussian.gif", 'pillow', fps=30)

    print("Test Complete")

    pass

def thin_offset_gaussian_test():
    print("Rendering thin offset gaussian test...")

    sim = SoftSimulator(delta_t=0.01, steps_per_frame=2, length=10)

    sim.set_psi(np.exp((-(sim.discrete_grid-2.0)**2), dtype=complex))

    anim = sim.animate_momentum(400, plot_func=SoftSimulator.probability_density, plot_V=True, pos_ylim=(0, 1))
    anim.save("SOFT Thin Offset Gaussian.gif", 'pillow', fps=30)

    print("Test Complete")

    pass

def free_particle_test():
    print("Rendering free particle test...")

    def free(x: np.ndarray):
        return 0 * x

    system = SoftSimulator(potential_energy_function=free, delta_t=0.1, steps_per_frame=2, length=30)

    momentum = np.exp(-((system.momentum_grid-(system.kmax/50))**2)*2, dtype=complex)
    # Set some momentum

    position = system.momentum_to_position(momentum) # + np.exp(-(system.discrete_grid + 1)**2)

    system.set_psi(position)

    anim = system.animate_momentum(400, plot_func=SoftSimulator.probability_density, plot_V=True, pos_ylim=(0, 0.5))
    anim.save("Free Particle Test.gif", 'pillow', fps=30)

    print("Test complete.")

    pass

def simple_coulomb_test():
    print("Rendering Simple Coulomb Test...")

    # We need to pre-declare all these variables so we can use them
    # To define the coulomb potential that aligns with the simulator.
    length = 20
    n = 11

    x_grid = np.linspace(-length/2, length/2, 2**n, endpoint=False)
    dx = length / 2**n

    def normalise(psi: np.ndarray) -> np.ndarray:
        return psi / (np.sqrt(np.vdot(psi, psi) * dx))

    vmax = 0.1*dx

    def coulomb_potential(x: np.ndarray) -> np.ndarray:
        return -1 / (np.abs(x) + vmax)

    sim = SoftSimulator(n_qubits=n, potential_energy_function = coulomb_potential, length = length, delta_t = 0.001, steps_per_frame=50)

    sim.set_psi(np.exp(-(sim.discrete_grid-1.0) ** 2 / 2, dtype=complex))

    anim = sim.animate(8000, plot_func=SoftSimulator.probability_density, plot_V=True, pos_ylim=(-1.0, 1.0))
    anim.save("Simple Coulomb Test.gif", 'pillow', fps=30)

    print("Test complete.")
    pass

def vary_vmax_test():

    n = 4
    values = 10.0**np.linspace(-2, 1, n)
    
    print(f"Rendering Vmax Variation Simulation for:\n Vmax = {values}")

    anim = vary_parameter_animation((11, 20.0, 0.0, 0, 0, 0),
                                    3,
                                    values,
                                    ['blue', 'red', 'green', 'orange', 'yellow'],
                                    8000,
                                    plot_func=SoftSimulator.probability_density,
                                    plot_V = True,
                                    ylim=(-1.0,1.0))
    
    anim.save("Vary Vmax Centred.gif", 'pillow', fps=30)
    
    print("Finished rendering")

    pass

def vary_length_test():
    n = 5
    values = np.linspace(10, 100, n)
    
    print(f"Rendering Length Variation Simulation for:\n Length = {values}")

    anim = vary_parameter_animation((11, 20.0, 0.0, 1.0, 0, 0),
                                    1,
                                    values,
                                    ['blue', 'red', 'green', 'orange', 'yellow'],
                                    8000,
                                    plot_func=SoftSimulator.probability_density,
                                    plot_V = True,
                                    ylim=(-1.0,1.0))
    
    anim.save("Vary Length Centred.gif", 'pillow', fps=30)
    
    print("Finished rendering")
    pass

def vary_dx_test():
    n = 5
    values = np.array([8, 9, 10, 11, 12])
    
    print(f"Rendering dx Variation Simulation for:\n dx = {20 / 2**values}")

    anim = vary_parameter_animation((11, 20.0, 0, 1.0, 0, 0),
                                    0,
                                    values,
                                    ['blue', 'red', 'green', 'orange', 'yellow'],
                                    8000,
                                    plot_func=SoftSimulator.probability_density,
                                    plot_V = True,
                                    ylim=(-1.0,1.0))
    
    anim.save("Vary dx Centred.gif", 'pillow', fps=30)
    
    print("Finished rendering")
    pass

def vary_dt_test():
    n = 4
    values = np.array([0.001, 0.0005, 0.0001, 0.00005])
    
    print(f"Rendering dt Variation Simulation for:\n dt = {values}")

    anim = vary_parameter_animation((11, 20.0, 0, 1.0, 0, 1000),
                                    4,
                                    values,
                                    ['blue', 'red', 'green', 'orange', 'yellow'],
                                    64000,
                                    plot_func=SoftSimulator.probability_density,
                                    plot_V = True,
                                    ylim=(-1.0,1.0))
    
    anim.save("Vary dt Offset.gif", 'pillow', fps=30)
    
    print("Finished rendering")
    pass

def vmax_to_zero():
    n = 4
    values = np.array([1, 0.5, 0.000000000000000001, 0])

    print(f"Rendering simulation for Vmax tending to zero with values: {values}")

    anim = vary_parameter_animation((11, 20, 0, 0, 0, 50),
                                    3,
                                    values,
                                    ['red', 'orange', 'yellow', 'green', 'blue'],
                                    8000,
                                    plot_func=SoftSimulator.probability_density,
                                    plot_V = True,
                                    ylim=(-1.0, 1.0))
    
    anim.save("Vmax to zero.gif", 'pillow', fps=30)

    pass

def energy_calc_test():
    sim = SoftSimulator(steps_per_frame=1, length = 10)
    
    num_frames = 400


    fig = plt.figure()
    axis = plt.axes(xlim=(0, num_frames*sim.steps_per_frame))
    xdat = np.linspace(0, num_frames*sim.steps_per_frame, num_frames, endpoint=False)
    ydat = []

    energy_line, = axis.plot([], [], c='blue', lw=1)

    for frame_num in range(num_frames):
        ydat.append(np.real(sim.get_energy()))
        for j in range(sim.steps_per_frame):
            sim.next_time_step()

    axis.plot(xdat[1:], ydat[1:], lw=1, color='blue')
    plt.show()

    # sim = SoftSimulator(steps_per_frame=2, length = 10, delta_t = 0.01)
    # anim = sim.animate(num_frames, plot_func=SoftSimulator.probability_density, plot_V=True, pos_ylim=(0, 1.0), text_E=True)
    # anim.save("Energy Shown Test.gif", 'pillow', fps=30)

    print("Did the maths for you.")

    pass

def energy_vary_vmax():
    sim = SoftSimulator(steps_per_frame=1, length = 10)
    
    # 10 qubits, length = 10, dx, vmax, delta_t, steps_per_frame
    params = (10, 10, 0, 1.0, 0.01, 1)

    num_frames = 400

    n = 4
    values = 10.0**np.linspace(-5, 1, n)
    
    print(f"Calculating Energy Change for Various Vmax:\n Vmax = {values}")

    colors=['blue', 'green', 'yellow', 'red']
    sims = []

    for i in range(n):
        temp_param = list(params)
        temp_param[3] = values[i]
        temp_param = tuple(temp_param)
        sims.append(generate_coulomb_simulator(temp_param))

    fig = plt.figure()
    axis = plt.axes(xlim=(0, num_frames*sims[0].steps_per_frame))
    xdat = np.linspace(0, num_frames*sims[0].steps_per_frame, num_frames, endpoint=False)
    ydat = []

    energy_lines = []
    for i in range(n):
            lobj, = axis.plot([], [], c=colors[i], lw=1, label=f'{values[i]}')
            energy_lines.append(lobj)
            ydat.append([])

    for frame_num in range(num_frames):
        for i in range(n):
            ydat[i].append(np.real(sims[i].get_energy()))
            for j in range(sims[i].steps_per_frame):
                sims[i].next_time_step()

    for i in range(n):
        axis.plot(xdat[1:], ydat[i][1:], lw=1, color=colors[i])
    
    plt.legend()

    plt.show()
    pass

def main():
    print("Welcome to the simulator!")

    # vary_vmax_test()
    # vary_length_test()
    # vmax_to_zero()

    # energy_vary_vmax()
    energy_calc_test()


    print("Simulation finished and saved.")
    pass
    
if __name__=="__main__":
    main()
