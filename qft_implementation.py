import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image
from qutip import *
from qutip_qip.operations import *
from qutip_qip.circuit import QubitCircuit, Gate


class QFT:
    def __init__(self, N=4):
        self.N = N
        self.qc = QubitCircuit(self.N)

        for i in range(N):
            self.qc.add_gate("SNOT", targets=[i])
            for j in range(1, N-i):
                # Add controlled phase rotation
                k = (j+1)
                phase_change = 2 * np.pi / (2**(k))
                # label = r'{\frac{{2 \pi}}{{ 2^{{ {num} }}  }}}'.format(num=k)
                label = r'{num}'.format(num=k)
                self.qc.add_gate("CPHASE", controls=[j+i], targets=[i], arg_value=phase_change, arg_label=label)

        pass
    
    def run(self, state: Qobj):
        return self.qc.run(state=state)

    def plot(self):
        return self.qc.png
    
    def save_circuit_diagram(self, filename: str):
        with open(filename, "wb") as out:
            out.write(self.plot().data)
        pass

N=2
qft = QFT(N)

# qft.save_circuit_diagram("Output.png")

i = 1
state = tensor(basis(2, (i & (1 << 1)) >> 1), 
                basis(2, (i & (1 << 0)) >> 0))#, 
                # basis(2, (i & (1 << 2)) >> 2), 
                # basis(2, (i & (1 << 3)) >> 3))
print(state)
print(qft.run(state))

# Manually check that i=1 is correct.

# b = Bloch()
# b.make_sphere()

# zeroth_qubit = basis(2, 0) + 1j * basis(2,1)
# zeroth_qubit /= np.sqrt(2)
# first_qubit = basis(2,0) - basis(2,1)
# first_qubit /= np.sqrt(2)

# b.add_states(zeroth_qubit)
# b.add_states(first_qubit)
# b.render()
# plt.show()