# Let's visualize how a single Fourier mode evolves in an expanding universe
# We'll solve the mode equation:
#   a_k'' + 3H a_k' + (k^2/a(t)^2) a_k = 0
# for radiation expansion

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("TkAgg")   # or Qt5Agg if installed

N = 5
# Parameters
H = 1.0          # Hubble parameter
k = 5.0          # comoving wavenumber
kappa = np.logspace(-1, 1, N)
ma = 1e-5
t_min, t_max = 0, 4
t_eval = np.linspace(t_min, t_max, 2000)

# Scale factor
def R(t):
    return np.sqrt(1/(t + 1e-4))

# Axion mass
def m(t):
    return (t + 1e-4)**(2)

# Differential equation system
def mode_equation(t, y, H, k):
    ak, ak_dot = y
    omega2 = (k**2) / (R(t)**2) + ma**2
    return [ak_dot, -3*H*ak_dot - omega2*ak]

# Initial conditions: oscillatory regime (subhorizon)
y0 = [1.0, 0.0]
print('ok1')
sol = []
for i in range(N):
    solution = solve_ivp(mode_equation, [t_min, t_max], y0, t_eval=t_eval, args=[1.0, kappa[i]])
    sol.append(solution)
    print('ok ', i)
# sol2 = solve_ivp(mode_equation, [t_min, t_max], y0, t_eval=t_eval, args=[1.0, 3.0])
# Plot the mode amplitude
# plt.figure(1)
# plt.grid()
# for i in range(N):
#     plt.plot(sol[i].t, sol[i].y[0], label=f"k = {kappa[i]} H")
# # plt.plot(sol1.t, sol1.y[0],'b', sol2.t, sol2.y[0],'r')
# plt.legend()
# plt.xlabel("Time t")
# plt.ylabel("Mode amplitude a_k(t)")
# plt.title("Evolution of a Single Fourier Mode in Radiation dominated Expansion")
# plt.show()

sol1=[]
twopi = 2*np.pi
theta = np.linspace(-twopi, twopi, 25)
for i in range(25):
    y0 = [theta[i], 0.0]
    solution = solve_ivp(mode_equation, [t_min, t_max], y0, t_eval=t_eval, args=[1.0, 2.0])
    sol1.append(solution)
    print('ok ', i)

plt.figure()
plt.grid()
for i in range(25):
    plt.plot(sol1[i].t, sol1[i].y[0], label=f"\theta = {theta[i]}")
# plt.plot(sol1.t, sol1.y[0],'b', sol2.t, sol2.y[0],'r')
# plt.legend()
plt.xlabel("Time t")
plt.ylabel("Mode amplitude a_k(t)")
plt.title("Evolution of a Single Fourier Mode in Radiation dominated Expansion")
plt.show()