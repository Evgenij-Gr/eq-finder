import numpy as np
import systems_fun as sf
import findTHeteroclinic as FH
import TwoPendulumsSystemFun as tpsf
import itertools as itls
from functools import partial
import multiprocessing as mp
from MySystem import *


class TwoOscillators:
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


ep = sf.EnvironmentParameters('C:/Users/User/eq-finder/output_files/Eq List', 'eq', 'Image')

boundsType = [(-0.1, 2*np.pi+0.1), (-0.1, 2*np.pi+0.1)]
bordersType = [(-1e-15, +2 * np.pi + 1e-15), (-1e-15, +2 * np.pi + 1e-15)]

# paramK = 0.06
# N = 5
# M = 5
# Gamma = np.linspace(0., 1.5, N)
# Lambda = np.linspace(0., 1.5, M)
# #np.savetxt('someFile.txt', [[], []], fmt='%+18.15f')
#
# i = 0
# for paramGamma in Gamma:
#     j = 0
#     for paramLambda in Lambda:
#
#         Sys = TwoOscillators(paramGamma, paramLambda, paramK)
#         TestJacType = Sys.JacType
#         TestRhsType = Sys.ReducedSystem
#         TestRhs = Sys.FullSystem
#         Eq = sf.findEquilibria(TestRhsType, TestJacType, boundsType, bordersType, sf.ShgoEqFinder(300, 30, 1e-10), sf.STD_PRECISION)
#
#         for eq in Eq:
#             eq.coordinates = mapBackTo4D(eq.coordinates)
#
#         sf.writeToFileEqList(ep, Eq, [paramGamma, paramLambda], "{:0>5}_{:0>5}".format(i, j), sf.STD_PRECISION)
#         j+=1
#     i+= 1
# sf.createBifurcationDiag(ep, N, M, Gamma, Lambda)


def parallEqList(params, paramK):
    (i, Gamma), (j, Lambda) = params
    Sys = TwoPendulums(Gamma, Lambda, paramK)
    rhs = Sys.FullSystem
    jacType = Sys.JacType
    rhsType = Sys.ReducedSystem
    rhsJac = Sys.Jac
    Eq = sf.findEquilibria(rhs, rhsJac, rhsType, jacType, mapBackTo4D, boundsType, bordersType,
                           sf.ShgoEqFinder(300, 30, 1e-10), sf.STD_PRECISION)

    sf.writeToFileEqList(ep, Eq, [Gamma, Lambda], "{:0>5}_{:0>5}".format(i, j), sf.STD_PRECISION)


if __name__ == "__main__":
    configFile = open('C:/Users/User/eq-finder/config.txt', 'r')
    configDict = eval(configFile.read())
    N, M, gammas, lambdas, paramK = tpsf.get_grid(configDict)
    pool = mp.Pool(mp.cpu_count())
    pool.map(partial(parallEqList, paramK=paramK),
                   itls.product(enumerate(gammas), enumerate(lambdas)))
    pool.close()

    sf.createBifurcationDiag(ep, N, M, lambdas, gammas)