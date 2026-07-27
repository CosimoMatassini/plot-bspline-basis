import matplotlib.pyplot as plt
import numpy as np
from bsplines import plot_bspline_basis
from scipy.interpolate import BSpline


def plot_tp_bases(degree_u, degree_v, knotU, knotV):

    n_basis_u = len(knotU) - degree_u - 1
    n_basis_v = len(knotV) - degree_v - 1

    plot_bspline_basis(knotU, degree_u)
    plot_bspline_basis(knotV, degree_v)

    u = np.linspace(knotU[degree_u], knotU[-degree_u - 1], 100)
    v = np.linspace(knotV[degree_v], knotV[-degree_v - 1], 100)

    Nu = []
    for i in range(n_basis_u):
        coeff = np.zeros(n_basis_u)
        coeff[i] = 1
        Nu.append(BSpline(knotU, coeff, degree_u)(u))
    Nu = np.array(Nu)

    Mv = []
    for j in range(n_basis_v):
        coeff = np.zeros(n_basis_v)
        coeff[j] = 1
        Mv.append(BSpline(knotV, coeff, degree_v)(v))
    Mv = np.array(Mv)

    U, V = np.meshgrid(u, v)

    fig = plt.figure()

    for i in range(n_basis_u):
        for j in range(n_basis_v):
            ax = fig.add_subplot(
                n_basis_u, n_basis_v, i * n_basis_v + j + 1, projection="3d"
            )

            Z = np.outer(Mv[j], Nu[i])

            ax.plot_surface(U, V, Z, linewidth=0, antialiased=True)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.view_init(elev=35, azim=-50)

    plt.savefig("tensor_product_bspline_bases.pdf", dpi=300, bbox_inches="tight")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    degree_u = 2
    degree_v = 2
    knotU = np.array([0, 0, 0, 1.75, 7, 11], dtype=float)
    knotV = np.array([0, 0, 1, 2, 3, 4, 4], dtype=float)
    plot_tp_bases(degree_u, degree_v, knotU, knotV)
