import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Set the size of the grid and the maximum number of iterations
N = 50
max_iter = 100

# Initialize the grid with random values
grid = np.random.randint(2, size=(N, N))

# Define the function to update the grid at each iteration
def update(frameNum, img, grid, N):
    # Copy the grid to avoid overwriting
    newGrid = grid.copy()
    # Iterate through each cell
    for i in range(N):
        for j in range(N):
            # Compute the sum of the neighbors' values
            neighbors = (grid[i, (j-1)%N] + grid[i, (j+1)%N] +
                         grid[(i-1)%N, j] + grid[(i+1)%N, j] +
                         grid[(i-1)%N, (j-1)%N] + grid[(i-1)%N, (j+1)%N] +
                         grid[(i+1)%N, (j-1)%N] + grid[(i+1)%N, (j+1)%N])
            # Apply the rules of the game
            if grid[i, j] == 0 and neighbors == 3:
                newGrid[i, j] = 1
            elif grid[i, j] == 1 and (neighbors < 2 or neighbors > 3):
                newGrid[i, j] = 0
    # Update the grid
    img.set_data(newGrid)
    grid[:] = newGrid[:]
    return img,

# Create the animation
fig, ax = plt.subplots()
img = ax.imshow(grid, interpolation='nearest', cmap='binary')
ani = animation.FuncAnimation(fig, update, frames=max_iter, fargs=(img, grid, N), interval=50, blit=True)

plt.show()
