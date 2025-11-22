import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline


def plot_bspline_basis(knots, degree):
    knots = np.asarray(knots)

    n_basis = len(knots) - 1 - degree

    if n_basis < 1:
        print("Error: Invalid input.")
        print(f"Number of basis functions ({n_basis}) must be >= 1.")
        print("Check if len(knots) > degree + 1.")
        return
    x_plot = np.linspace(knots[0], knots[-1], 500)

    for i in range(n_basis):
        # The i-th basis function N_{i,k} is defined by the knots
        # t[i], ..., t[i+k+1]
        knots_i = knots[i : i + degree + 2]

        # Create the B-spline object for *only* the i-th basis function
        # This function is non-zero only on [knots[i], knots[i+k+1]]
        spl = BSpline.basis_element(knots_i, extrapolate=False)

        # Evaluate the basis function over the full x_plot range
        # spl(x_plot) will be NaN outside its support
        y_plot = spl(x_plot)

        # Plot, replacing NaNs (outside the support) with 0
        plt.plot(
            x_plot, np.nan_to_num(y_plot), label=f"$N_{{{i},{degree + 1}}}(x)$", lw=2.5
        )

    plt.title(f"B-Spline Basis Functions (Degree {degree})", fontsize=16)

    # Add vertical lines for unique knots to show their influence
    unique_knots = np.unique(knots)

    plt.plot(knots[degree], 0, "ro", markersize=8)
    plt.plot(knots[n_basis], 0, "ro", markersize=8)

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
    d = 2
    t = [0, 0, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3, 3]

    plot_bspline_basis(t, d)
