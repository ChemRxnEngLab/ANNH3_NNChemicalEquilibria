import numpy as np


def fill_out(x):
    return np.array([x[0], x[1], 1 - x[0] - x[1]])


def calc_n_eq(x, n, x_eq):
    # A*x=b
    b = x * n
    A = np.eye(3) - x_eq.reshape(-1, 1)
    return x * n + np.linalg.solve(A, b)


def calc_mb_error(n, n_eq):
    del_n = n - n_eq
    ESM = np.array(
        [  # N #H
            [0, 2],  # H2
            [1, 3],  # NH3
            [2, 0],  # N2
        ]
    )
    return np.sqrt(np.sum((del_n @ ESM) ** 2))
