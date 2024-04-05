""" This code solves the Schrödinger equation in terms of y for potential 
    V(y) = V_0(y^2/C^2-2*y/C) for energy eigenvalue E and then plots the wavefunction
    in terms of y and x = - a*log(y) along with the potential.
    
    Finding dimensionless wavefunction i.e. hbar = 1, m = 1
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

V0 = 1
C = 1
a = 1


# Defining the potential
def potential(y):
    return V0 * (y ** 2 - 2 * y) / C

# Defining the Schrödinger equation
def s_eq(y, psi, E):
    return [psi[1], -psi[1] / y - 2 * a ** 2 * (E / y ** 2 - potential(y) / y ** 2) * psi[0]]


# Defining shooting method to solve for eigenstate with energy eigenvalue between E = 0 and E = 3.0
def shooting_method(a, b, alpha, beta, tol=1e-6, max_iter=1000):
    def residual(guess):
        sol = solve_ivp(lambda y, psi: s_eq(y, psi, guess), [a, b], [alpha, guess], t_eval=[b], method='RK45')
        return sol.y[0][-1] - beta  # Finding the differernce between psi(b) for E = guess and beta

    guess_min = 0.0  # initial guess for the shooting parameter Energy eigenvalue
    guess_max = 3.0
    iter_count = 0

    while iter_count < max_iter:
        guess = (guess_min + guess_max) / 2.0
        res = residual(guess)

        if abs(res) < tol:
            break
        elif res < 0:
            guess_max = guess
        else:
            guess_min = guess

        iter_count += 1

    if iter_count == max_iter:
        print("Maximum iterations reached without convergence")

    return solve_ivp(lambda y, psi: s_eq(y, psi, guess), [a, b], [alpha, guess], method='RK45', t_eval=np.linspace(a, b, 3000))


# Solve the Schrödinger equation using shooting method
solution = shooting_method(0.001, 10, -1.0, 0.0)  # psi(0.001) = -1.0 and psi(10) = 0.0

# Plotting the solution
plt.plot(solution.t, solution.y[0], label='Eigenfunction $\psi(y)$')
plt.plot(solution.t, potential(solution.t), 'r', label='Potential $V(y)$')
plt.xlim(-0.1,7.5)
plt.ylim(-2.0, 5.0)
plt.xlabel('y')
plt.ylabel('$\psi(y)$, $V(y)$')
plt.title('Energy Eigenfunction for Schrödinger Equation')
plt.legend()
plt.grid()
plt.show()

plt.plot(-a * np.log(solution.t), solution.y[0], 'black', label='$\Psi(x)$')
plt.plot(-a * np.log(solution.t), potential(solution.t), 'red', label='V(x)')
plt.ylim(-7.0, 20.0)
plt.xlabel('x')
plt.ylabel('$\Psi(x)$,$V(x)$')
plt.title('Solution of the Schrodinger Equation for V(x = -a ln(y))')
plt.legend()
plt.grid()
plt.show()
