import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from matplotlib.animation import FuncAnimation

# Constants
Nx = 2000
Nt = 1000
scal = 10
x_1 = np.linspace(-15, 15, Nx)
dx = x_1[1] - x_1[0]

# Input
print('Input the states that the system needs to be in:')
state_1 = int(input("State 1: "))
state_2 = int(input("State 2: "))
print(f'state one is {state_1} \nstate two is {state_2}')

# Potential
def V(x):
    return np.where(np.abs(x) <= scal, 0.5 * x**2, 0.5 * scal**2)

# Wavefunction class
class psi:
    def __init__(self, state):
        self.state = state
        kinetic = diags([1, -2, 1], offsets=[-1, 0, 1], shape=(Nx, Nx)).toarray() / dx**2
        potential = np.diag(V(x_1))
        H = -0.5 * kinetic + potential

        e_vals, e_vecs = np.linalg.eigh(H)
        psi_x = e_vecs[:, state]

        norm = np.sqrt(np.sum(np.abs(psi_x)**2) * dx)
        psi_x = psi_x / norm
        psi_x = psi_x[::-1]
        self.psi_x = psi_x
        self.E = e_vals[state]

# Get wavefunctions
PSI_1 = psi(state_1)
PSI_2 = psi(state_2)

# Superposition (complex form)
PSI_tot = (PSI_1.psi_x + PSI_2.psi_x) / np.sqrt(2)

# Normalize
PSI_tot /= np.sqrt(np.sum(np.abs(PSI_tot)**2) * dx)

# Real parts for plotting
PSI_list_real = np.array([PSI_1.psi_x.real, PSI_2.psi_x.real, PSI_tot.real])
name_ = np.array(['state one', 'state two', 'superposition'])
color_ = np.array(['blue', 'red', 'green'])
states_ = np.array([state_1, state_2, f'States {state_1} and {state_2}'])

# Plot psi(x)
plt.figure(figsize=(18, 5))
for i in range(len(name_)):
    plt.subplot(1, 3, i + 1)
    plt.title(r'$\psi(x,0)$', fontsize=32)
    plt.plot(x_1, PSI_list_real[i], label=name_[i] + ' of ' + states_[i], color=color_[i])
    plt.xlabel('x', fontsize=24)
    plt.ylabel(r'$\psi(x)$', fontsize=24)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

# Plot |psi(x)|^2
plt.figure(figsize=(18, 5))
for i in range(len(name_)):
    plt.subplot(1, 3, i + 1)
    plt.title(r'$|\psi(x,0)|^2$', fontsize=32)
    plt.plot(x_1, np.abs(PSI_list_real[i])**2, label=name_[i] + ' of ' + states_[i], color=color_[i])
    plt.xlabel('x', fontsize=24)
    plt.ylabel(r'$|\psi(x)|^2$', fontsize=24)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

plt.show(block=True)

# Normalization checks
print(f"Normalization of state 1: {np.sum(np.abs(PSI_1.psi_x)**2) * dx:.6f}")
print(f"Normalization of state 2: {np.sum(np.abs(PSI_2.psi_x)**2) * dx:.6f}")
print(f"Normalization of superposition: {np.sum(np.abs(PSI_tot)**2) * dx:.6f}")

# Animation
def animate_wavefunction():
    global ani

    t = np.linspace(0, 2 * np.pi / np.abs(PSI_1.E - PSI_2.E), Nt)

    fig, ax = plt.subplots(figsize=(18, 5))
    line, = ax.plot([], [], lw=2, label=r'$|\psi(x,t)|^2$', color=color_[2])

    ax.set_xlim(x_1[0], x_1[-1])
    ax.set_ylim(0, 1.2 * np.max(np.abs(PSI_tot)**2))
    ax.set_xlabel("x", fontsize=18)
    ax.set_ylabel(r"$|\psi(x,t)|^2$", fontsize=18)
    ax.set_title(f'Time Evolution of Superpostion Wave\nof States {state_1} and {state_2}', fontsize=20)
    ax.grid()
    ax.legend()

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        t_now = t[frame]
        psi_t = (
            PSI_1.psi_x * np.exp(-1j * PSI_1.E * t_now) +
            PSI_2.psi_x * np.exp(-1j * PSI_2.E * t_now)
        ) / np.sqrt(2)

        prob_density = np.abs(psi_t)**2
        line.set_data(x_1, prob_density)
        return line,

    ani = FuncAnimation(fig, update, frames=Nt, init_func=init, blit=True, interval=5)
    plt.tight_layout()
    plt.show()

animate_wavefunction()
