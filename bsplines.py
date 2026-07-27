import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline


def plot_bspline_basis(knots, degree):

    n_basis = len(knots) - 1 - degree

    if n_basis < 1:
        print(
            f"Error: Knots are {len(knots)}. They must be more than {degree + 1} (degree + 1)"
        )
        return
    x_plot = np.linspace(knots[0], knots[-1], 500)

    for i in range(n_basis):
        knots_i = knots[i : i + degree + 2]
        spl = BSpline.basis_element(knots_i, extrapolate=False)
        y_plot = spl(x_plot)
        plt.plot(
            x_plot, np.nan_to_num(y_plot), label=f"$N_{{{i},{degree}}}(x)$", lw=2.5
        )

    plt.plot(knots[degree], 0, "ro", markersize=8)
    plt.plot(knots[n_basis], 0, "ro", markersize=8)

    unique_knots = np.unique(knots)
    for k in unique_knots:
        plt.axvline(k, color="gray", linestyle="--", linewidth=0.5)

    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    degree = 3
    knots = np.array([0, 0, 0, 2, 3, 3.5, 4, 5, 5, 5])
    plot_bspline_basis(knots, degree)
