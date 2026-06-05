import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BSpline


def plot_bspline_basis(n_intervals, degree):

    knots = np.concatenate(
        (
            np.zeros(degree),
            np.linspace(0, 1, n_intervals + 1),
            np.ones(degree),
        )
    )

    n_basis = len(knots) - 1 - degree

    if n_basis < 1:
        print("Error: Number of basis functions must be positive!")
        print("Check if number of number of knots is greater than degree + 1")
        return
    x_plot = np.linspace(knots[0], knots[-1], 500)

    for i in range(n_basis):
        knots_i = knots[i : i + degree + 2]
        spl = BSpline.basis_element(knots_i, extrapolate=False)
        y_plot = spl(x_plot)
        plt.plot(
            x_plot, np.nan_to_num(y_plot), label=f"$N_{{{i},{degree + 1}}}(x)$", lw=2.5
        )

    plt.title(f"B-Spline Basis Functions (Degree {degree})", fontsize=16)

    unique_knots = np.unique(knots)

    for k in unique_knots:
        plt.axvline(k, color="gray", linestyle="--", linewidth=0.8)

    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.ylabel("Basis Function Value", fontsize=12)
    plt.xlabel("x", fontsize=12)
    plt.figtext(
        0.5,
        0.02,
        f"Knot vector: {knots.tolist()}",
        ha="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
    )
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    d = 8
    n_intervals = 5
    plot_bspline_basis(n_intervals, d)
