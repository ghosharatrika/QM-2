""" This code solves the Schrödinger equation in terms of y for potential 
    V(y) = V_0(y^2/C^2-2*y/C) for energy eigenvalue E and then plots the wavefunction
    in terms of y and x = - a*log(y) along with the potential"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

V0 = 1
C = 1
E = 0.01  # Energy eigenvalue
a = 1


def potential(y, C, V0):  # Defining the potential
    return V0 * (y ** 2 - 2 * y) / C


# Defining the Schrödinger equation
def model(wf, y, E, V0, C, a):
    psi, dpsi = wf
    d2psi = [dpsi, -dpsi / y - 2 * a ** 2 * (E / y ** 2 - potential(y, C, V0) / y ** 2) * psi]
    return d2psi


# Initial conditions
psi0 = [-1.0, 1.0]  # psi(y0) = -1.0, psi'(y0) = 0

# Space points
y = np.linspace(0.001, 4, 2000)  # from 0.001 to 4, 2000 points
x = -np.log(y)  # Converting y to x to plot psi(x)

psi1 = odeint(model, psi0, y, args=(E, V0, C, a))  # Solving the Schrödinger equation
# Plot the solution
plt.plot(y, psi1[:, 0], 'black', label='Psi(y)')
plt.plot(y, potential(y, C, V0), 'red', label='V(y)')
plt.ylim(-1.2, 1.0)
plt.xlabel('y')
plt.ylabel('Psi(y),V(y)')
plt.title('Solution of the Schrodinger Equation for V(y)')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(x, psi1[:, 0], 'b', label='Psi(x)')
plt.plot(x, potential(y, C, V0), 'r', label='V(x)')
plt.ylim(-1.5, 2.5)
plt.xlabel('x')
plt.ylabel('Psi,V(x)')
plt.title('Solution of the Schrodinger Equation for V(x = -a*ln(y))')
plt.legend()
plt.grid()
plt.show()


