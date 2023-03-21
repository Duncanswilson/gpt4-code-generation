import numpy as np
import control

def lqr(A, B, Q, R):
    """
    Compute the optimal gain matrix K for a discrete-time LQR controller.
    
    Args:
    A: System state matrix (n x n)
    B: Input matrix (n x m)
    Q: State cost matrix (n x n)
    R: Input cost matrix (m x m)
    
    Returns:
    K: Optimal gain matrix (m x n)
    """
    # Solve the discrete-time Algebraic Riccati Equation
    P = control.dare(A, B, Q, R)
        
    print(P)
    print(P[0])

    # Compute the optimal gain matrix K
    K = np.linalg.inv(R + B.T @ P[0] @ B) @ (B.T @ P[0] @ A)
    
    return K

# Define your system matrices (example)
A = np.array([[1.0, 1.0], [0.0, 1.0]])
B = np.array([[0.5], [1.0]])
Q = np.eye(2)  # State cost matrix
R = np.array([[1]])  # Input cost matrix

# Compute the optimal gain matrix K
K = lqr(A, B, Q, R)
print("Optimal gain matrix K:", K)

# Simulate the closed-loop system
x0 = np.array([[-1.0], [1.0]])  # Initial condition
u0 = -K @ x0  # Initial control input
print("Initial control input:", u0)
