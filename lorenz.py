import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the Lorenz system of differential equations
def lorenz(t, xyz, sigma, beta, rho):
    x, y, z = xyz
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return [dx_dt, dy_dt, dz_dt]

# Set parameters
sigma = 10
beta = 8/3
rho = 28

# Initial conditions
xyz0 = [0.4, 0.1, 0.8]

# Solve the system of differential equations
solution = solve_ivp(lorenz, [0, 100], xyz0, args=(sigma, beta, rho), dense_output=True)

# Extract the solution
xyz = solution.sol(np.linspace(0, 100, 10000))

# Plot the solution in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(xyz[0], xyz[1], xyz[2], color='b', lw=0.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Lorenz System')
plt.show()
