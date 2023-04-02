import numpy as np
import matplotlib.pyplot as plt

# Set the size of the grid and the maximum number of iterations
N = 1000
max_iter = 100

# Define the range of x and y values
x_min, x_max = -2, 2
y_min, y_max = -2, 2

# Create a 2D grid of complex numbers
x = np.linspace(x_min, x_max, N)
y = np.linspace(y_min, y_max, N)
X, Y = np.meshgrid(x, y)
c = X + 1j*Y

# Initialize the array for the Mandelbrot set
Z = np.zeros((N, N), dtype=np.complex128)

# Iterate the complex equation
for i in range(max_iter):
    Z = Z**2 + c

# Compute the absolute value of the complex number
abs_Z = np.abs(Z)

# Plot the Mandelbrot set
fig = plt.figure(figsize=(10, 10))
plt.imshow(abs_Z, cmap='hot', extent=[x_min, x_max, y_min, y_max])
plt.title('Mandelbrot Set')
plt.show()
