import numpy as np
import systems_fun as sf
import findTHeteroclinic as FH
import TwoPendulumsSystemFun as tpsf
import itertools as itls
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.optimize import root


class TwoPendulums:
    def __init__(self, Gamma, Lambda, paramK):
        self.Gamma = Gamma
        self.Lambda = Lambda
        self.paramK = paramK

    def FullSystem(self, X):
        fi1, V1, fi2, V2 = X
        return [V1,
                self.Gamma - self.Lambda * V1 - np.sin(fi1) + self.paramK * np.sin(fi2 - fi1),
                V2,
                self.Gamma - self.Lambda * V2 - np.sin(fi2) + self.paramK * np.sin(fi1 - fi2)]

    def Jac(self, X):
        fi1, V1, fi2, V2 = X
        return[[0, 1, 0, 0],
               [-np.cos(fi1) - self.paramK * np.cos(fi2 - fi1), -self.Lambda, self.paramK * np.cos(fi2 - fi1), 0],
               [0, 0, 0, 1],
               [self.paramK * np.cos(fi1 - fi2), 0, -np.cos(fi2) - self.paramK * np.cos(fi1 - fi2), -self.Lambda]]

    def JacType(self, fis):
        fi1, fi2 = fis
        X = [fi1, 0., fi2, 0.]
        return self.Jac(X)

    def ReducedSystem(self, fis):
        fi1, fi2 = fis
        return [self.Gamma - np.sin(fi1) + self.paramK * np.sin(fi2 - fi1),
                self.Gamma - np.sin(fi2) + self.paramK * np.sin(fi1 - fi2)]


def mapBackTo4D(fis):
    fi1, fi2 = fis
    return [fi1, 0., fi2, 0.]

boundsType = [(-0.1, 2 * np.pi + 0.1), (-0.1, 2 * np.pi + 0.1)]
bordersType = [(-1e-15, +2 * np.pi + 1e-15), (-1e-15, +2 * np.pi + 1e-15)]


# if __name__ == "__main__":
#     # Gamma = 0.75
#     # Lambda = 0.6595306
#     Gamma = Lambda = np.linspace(0., 1., 11)
#     paramK = 0.
#     grid = np.zeros([len(Gamma), len(Lambda)])
#     for i, y in enumerate(Lambda):
#         for j, x in enumerate(Gamma):
#             print(j, i)
#             Sys = TwoPendulums(x, y, paramK)
#             rhs = Sys.FullSystem
#             jacType = Sys.JacType
#             rhsType = Sys.ReducedSystem
#             rhsJac = Sys.Jac
#             Eq = sf.findEquilibria(rhs, rhsJac, rhsType, jacType, mapBackTo4D, boundsType, bordersType,
#                                    sf.ShgoEqFinder(300, 30, 1e-10), sf.STD_PRECISION)
#
#             newEq = [eq for eq in Eq if sf.is4DSaddleFocusWith1dU(eq, sf.STD_PRECISION)]
#             newEq = [eq for eq in newEq if eq.coordinates[0] < eq.coordinates[2]]
#
#             for eq in newEq:
#                 # print("######")
#                 # print(f"Coords: {eq.coordinates}")
#                 # print(f"Eigenvals: {eq.eigenvalues}")
#                 # print(f"Type: {eq.getEqType(sf.STD_PRECISION)}")
#                 st = eq.getLeadSEigRe(sf.STD_PRECISION)
#                 # print(st)
#                 unst = eq.getLeadUEigRe(sf.STD_PRECISION)
#                 # print(unst)
#                 # print(f'Седловая величина б = {unst - (-1*st)}')
#                 # print(f'Седловой индекс р = {-st/unst}')
#
#                 grid[i][j] = -st/unst
#     plt.pcolormesh(Gamma, Lambda, grid, cmap=plt.cm.get_cmap('RdBu'))
#     # plt.axes().set_aspect('equal', adjustable='box')
#     plt.colorbar()
#     plt.xlabel('Gamma')
#     plt.ylabel('Lambda')
#     plt.title('Седловой индекс')
#     plt.show()

# Gamma = 0.5
# Lambda = 0.3
# paramK = 0.06
# Sys = TwoPendulums(Gamma, Lambda, paramK)
# rhs = Sys.FullSystem
# jacType = Sys.JacType
# rhsType = Sys.ReducedSystem
# rhsJac = Sys.Jac
# Eq = sf.findEquilibria(rhs, rhsJac, rhsType, jacType, mapBackTo4D, boundsType, bordersType,
#                                    sf.ShgoEqFinder(300, 30, 1e-10), sf.STD_PRECISION)
# for eq in Eq:
#     print("######")
#     print(f"Coords: {eq.coordinates}")
#     print(f"Eigenvals: {eq.eigenvalues}")
#     print(f"Type: {eq.getEqType(sf.STD_PRECISION)}")
#
#
# newEq = [eq for eq in Eq if sf.is4DSaddleFocusWith1dU(eq, sf.STD_PRECISION)]
# newEq = [eq for eq in newEq if eq.coordinates[0] < eq.coordinates[2]]
# for eq in newEq:
#     st = eq.getLeadSEigRe(sf.STD_PRECISION)
#     # print(st)
#     unst = eq.getLeadUEigRe(sf.STD_PRECISION)
#     sigma = unst + st
#     rho = -st / unst
#     print(rho)