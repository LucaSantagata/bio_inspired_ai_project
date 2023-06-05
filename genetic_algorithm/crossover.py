import numpy as np
from typing import Tuple


def single_point_binary_crossover(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    offspring1 = parent1.copy()
    offspring2 = parent2.copy()

    _, cols = parent2.shape
    col = np.random.randint(0, cols)

    offspring1[:, :col] = parent2[:, :col]
    offspring2[:, :col] = parent1[:, :col]

    return offspring1, offspring2


def single_point_binary_crossover_rc(parent1: np.ndarray, parent2: np.ndarray, major='r') -> Tuple[np.ndarray, np.ndarray]:
    offspring1 = parent1.copy()
    offspring2 = parent2.copy()

    rows, cols = parent2.shape
    row = np.random.randint(0, rows)
    col = np.random.randint(0, cols)

    if major.lower() == 'r':
        offspring1[:row, :] = parent2[:row, :]
        offspring2[:row, :] = parent1[:row, :]

        offspring1[row, :col+1] = parent2[row, :col+1]
        offspring2[row, :col+1] = parent1[row, :col+1]
    elif major.lower() == 'c':
        offspring1[:, :col] = parent2[:, :col]
        offspring2[:, :col] = parent1[:, :col]

        offspring1[:row+1, col] = parent2[:row+1, col]
        offspring2[:row+1, col] = parent1[:row+1, col]

    return offspring1, offspring2