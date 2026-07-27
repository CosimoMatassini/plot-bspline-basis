import numpy as np
from bsplines import plot_bspline_basis


def plot_uniform_basis(n_intervals, degree):

    knots = np.concatenate(
        (
            np.zeros(degree),
            np.linspace(0, 1, n_intervals + 1),
            np.ones(degree),
        )
    )
    plot_bspline_basis(knots, degree)


if __name__ == "__main__":
    degree = 5
    n_intervals = 1
    plot_uniform_basis(n_intervals, degree)
