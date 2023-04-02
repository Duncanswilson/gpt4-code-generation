The Hofstadter butterfly is a fractal-like pattern that arises in the study of electrons in a magnetic field. Here's how you can generate a Hofstadter butterfly plot using Python and matplotlib:

```python
import numpy as np
import matplotlib.pyplot as plt

# Define the magnetic flux values
phi_vals = np.linspace(0, 1, 200)

# Define the energy levels
energy_levels = np.zeros((200, 200))

# Define the recurrence relation for the energy levels
def E(n, m, phi):
    if n == 0:
        return np.cos(2*np.pi*m*phi)
    elif n == 1:
        return np.cos(2*np.pi*m*phi)/(2-np.cos(2*np.pi*m*phi))
    else:
        return (np.cos(2*np.pi*m*phi)*E(n-1, m, phi) - E(n-2, m, phi))/(1-np.cos(2*np.pi*m*phi)*E(n-1, m, phi))

# Compute the energy levels
for m in range(1, 201):
    for n in range(0, 200):
        energy_levels[n, m-1] = E(n, m, phi_vals)

# Plot the energy levels
fig, ax = plt.subplots(figsize=(8,8))
ax.imshow(energy_levels, extent=[0, 1, -10, 10], aspect='auto', cmap='jet')
ax.set_xlabel('Magnetic flux')
ax.set_ylabel('Energy (meV)')
ax.set_title('Hofstadter butterfly')
plt.show()
```
