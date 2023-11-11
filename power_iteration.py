import numpy as np


def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps

    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    ### YOUR CODE HERE

    eigenvector = np.random.rand(data.shape[0])
    eigenvector = eigenvector / np.linalg.norm(eigenvector)

    # Iterate for num_steps
    for _ in range(num_steps):
        # Multiply the matrix by the eigenvector
        eigenvector = np.dot(data, eigenvector)

        # Normalize the eigenvector
        eigenvector = eigenvector / np.linalg.norm(eigenvector)

    # Compute the eigenvalue
    eigenvalue = np.dot(eigenvector, np.dot(data, eigenvector))

    return eigenvalue, eigenvector