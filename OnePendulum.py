import numpy as np
import systems_fun as sf
import MySystem as MS
import TwoPendulumsSystemFun as tpsf
import itertools as itls
from functools import partial
import multiprocessing as mp
import os
import warnings
warnings.filterwarnings("ignore")


class OnePendulum:
    def __init__(self, Gamma, Lambda):
        self.Gamma = Gamma
        self.Lambda = Lambda

    def FullSystem(self, X):
        [fi, V] = X
        return [V, self.Gamma - self.Lambda * V - np.sin(fi)]

    def Jac(self, X):
        [fi, V] = X
        return[[0, 1], [-np.cos(fi), -self.Lambda]]

    def JacType(self, fi):
        X = [fi[0], 0.]
        # print(X)
        return self.Jac(X)

    def ReducedSystem(self, fi):
        # print(fi)
        return [self.Gamma - np.sin(fi[0])]


def mapBackTo2D(fi):
    return [fi[0], 0.]


boundsType = [(-0.1, 2*np.pi+0.1), (-0.1, 2*np.pi+0.1)]
bordersType = [(-1e-15, 2*np.pi + 1e-15), (-1e-15, 2*np.pi + 1e-15)]


def parallEqList(params, paramK, ep):
    (i, Gamma), (j, Lambda) = params
    Sys = OnePendulum(Gamma, Lambda)
    rhs = Sys.FullSystem
    jac = Sys.Jac
    red_rhs = Sys.ReducedSystem
    red_jac = Sys.JacType

    Eq = sf.findEquilibria(rhs, jac, red_rhs, red_jac, mapBackTo2D, boundsType, bordersType,
                           sf.ShgoEqFinder(100, 5, 1e-10), sf.STD_PRECISION)

    for eq in Eq:
        eq.coordinates = MS.mapBackTo4D([eq.coordinates[0], eq.coordinates[0]])

    new_sys = MS.TwoPendulums(Gamma, Lambda, paramK)
    new_jac = new_sys.Jac

    new_Eq = []
    for eq in Eq:
        new_eq = sf.getEquilibriumInfo(eq.coordinates, new_jac)
        new_Eq.append(new_eq)

    sf.writeToFileEqList(ep, new_Eq, [Gamma, Lambda], "{:0>5}_{:0>5}".format(i, j), sf.STD_PRECISION)


# Gamma = 0.5
# Lambda = 0.5
# k = 0.06
# Sys = OnePendulum(Gamma, Lambda)
# rhs = Sys.FullSystem
# jac = Sys.Jac
# red_rhs = Sys.ReducedSystem
# red_jac = Sys.JacType
#
# Eq = sf.findEquilibria(rhs, jac, red_rhs, red_jac, mapBackTo2D, boundsType, bordersType, sf.ShgoEqFinder(100, 5, 1e-10), sf.STD_PRECISION)
# print("In 2D:")
# for eq in Eq:
#     eq.coordinates = MS.mapBackTo4D([eq.coordinates[0], eq.coordinates[0]])
#     print("######")
#     print(f"Coords: {eq.coordinates}")
#     print(f"Eigenvals: {eq.eigenvalues}")
#     print(f"Type: {eq.getEqType(sf.STD_PRECISION)}")
#
# new_sys = MS.TwoPendulums(Gamma, Lambda, k)
# new_jac = new_sys.Jac
# print("In 4D:")
# for eq in Eq:
#     new_eq = sf.getEquilibriumInfo(eq.coordinates, new_jac)
#     print("######")
#     print(f"Coords: {new_eq.coordinates}")
#     print(f"Eigenvals: {new_eq.eigenvalues}")
#     print(f"Type: {new_eq.getEqType(sf.STD_PRECISION)}")

K = [0.03]
if __name__ == "__main__":
    for k in K:
        path = f'C:/Users/User/eq-finder/output_files/Eq List/Синхронное подмногообразие/{k}'
        os.mkdir(path)
        ep = sf.EnvironmentParameters(f'C:/Users/User/eq-finder/output_files/Eq List/Синхронное подмногообразие/{k}', 'eq',
                                      'Image')
        configFile = open('C:/Users/User/eq-finder/config.txt', 'r')
        configDict = eval(configFile.read())
        N, M, gammas, lambdas, paramK = tpsf.get_grid(configDict)
        paramK = k
        pool = mp.Pool(mp.cpu_count())
        pool.map(partial(parallEqList, paramK=paramK, ep=ep),
                       itls.product(enumerate(gammas), enumerate(lambdas)))
        pool.close()

        sf.createBifurcationDiag(ep, N, M, gammas, lambdas)

# # for eq in Eq:
# #     eq.coordinates = mapBackTo2D(eq.coordinates)
#
# newEq = [eq for eq in Eq if sf.is2DSaddle(eq, sf.STD_PRECISION)]
# for eq in Eq:
#     eq.coordinates[1] = 0.
#     print("######")
#     print(f"Coords: {eq.coordinates}")
#     print(f"Eigenvals: {eq.eigenvalues}")
#     print(f"Type: {eq.getEqType(sf.STD_PRECISION)}")
#
# pairs_to_check = [[newEq[0], newEq[0]]]
# # print(pairs_to_check)
# cnctInfo = FH.checkSeparatrixConnection(pairs_to_check, sf.STD_PRECISION, sf.STD_PROXIMITY, rhs,
#                                         jac, sf.idTransform, sf.pickBothSeparatrices, sf.idListTransform,
#                                         sf.anyNumber, 10., 200., Distance2D, listEqCoords=None)
#
# for i in cnctInfo:
#     print(f"min_dist = {i['dist']}")
#     sep = i['stPt']
# print(len(sep))
#
# # for coord in sep[:, 0]:
# #     coord = normalize(coord)
#
# fig = plt.figure(figsize=(4, 4))
#
#
# ax1 = fig.add_subplot(111)
# ax1.plot(sep[:, 0], sep[:, 1], color='k')
# ax1.scatter(Eq[0].coordinates[0], Eq[0].coordinates[1], c='green', s=20, label='C')
# ax1.scatter(Eq[1].coordinates[0], Eq[1].coordinates[1], c='red', s=20, label='S')
# ax1.set_xlim([-5, 5])
# ax1.set_ylim([-5, 5])
# plt.title(f'$\gamma$={Gamma}, $\lambda$ = {Lambda}')
# plt.grid(True)
# plt.legend()
#
# pathToDir = 'C:/Users/User/eq-finder/output_files/OnePendulum'
# imageName = 'plot ' + f"{Gamma = }, {Lambda= }"
# fullOutputName = os.path.join(pathToDir, imageName + '.png')
# # plt.savefig(fullOutputName, dpi=300)
# plt.show()