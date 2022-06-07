import numpy as np
import systems_fun as sf
import findTHeteroclinic as FH
import TwoPendulumsSystemFun as tpsf


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


f = open('C:/Users/User/eq-finder/output_files/test/Протяжка по параметру.txt', 'w+')

#Gamma = 0.705
Lambda = 0.1
paramK = 0.06
Gammas = np.linspace(0.2, 0.3, 11)
# fi1 = np.linspace(-0.1, 2*np.pi+0.1)
# fi2 = np.linspace(-0.1, 2*np.pi+0.1)
# V1 = np.linspace(-0.01, 0.01)
# V2 = np.linspace(-0.01, 0.01)
for Gamma in Gammas:
    print(f'Gamma = {Gamma}')
    Sys = TwoPendulums(Gamma, Lambda, paramK)
    rhs = Sys.FullSystem
# jac = Sys.Jac
# Eq = sf.findEquilibria(rhs, jac, bounds, borders, sf.ShgoEqFinder(300, 30, 1e-10), sf.STD_PRECISION)
# for eq in Eq:
#     print("######")
#     print(f"Coords: {eq.coordinates}")
#     print(f"Eigenvals: {eq.eigenvalues}")
#     print(f"Type: {eq.getEqType(sf.STD_PRECISION)}")
# print('Coordinates with normal Jac')
# for eq in Eq:
#     print(Sys.FullSystem(eq.coordinates))
    boundsType = [(-0.1, 2*np.pi+0.1), (-0.1, 2*np.pi+0.1)]
    bordersType = [(-1e-15, +2 * np.pi + 1e-15), (-1e-15, +2 * np.pi + 1e-15)]
    jacType = Sys.JacType
    rhsType = Sys.ReducedSystem
    rhsJac = Sys.Jac
    Eq = sf.findEquilibria(rhsType, jacType, boundsType, bordersType, sf.ShgoEqFinder(300, 30, 1e-10), sf.STD_PRECISION)

    for eq in Eq:
        eq.coordinates = mapBackTo4D(eq.coordinates)

#print('Coordinates with fictitious Jac')
    for eq in Eq:
        if eq.getEqType(sf.STD_PRECISION) == [2, 0, 2, 0, 0]:
            f.write(f"###### \n Gamma = {Gamma} \n Coords: {eq.coordinates}\n Eigenvalues: {eq.eigenvalues} \n Type: {eq.getEqType(sf.STD_PRECISION)} \n")

        # print("######")
        # print(f"Coords: {eq.coordinates}")
        # print(f"Eigenvals: {eq.eigenvalues}")
        # print(f"Type: {eq.getEqType(sf.STD_PRECISION)}")
f.close()
# newEq = [eq for eq in Eq if sf.is4DSaddleFocusWith1dU(eq, sf.STD_PRECISION)]

# print('List with SaddleFocus eq:')
# for eq in newEq:
#     print("######")
#     print(f"Coords: {eq.coordinates}")
#     print(f"Eigenvals: {eq.eigenvalues}")
#     print(f"Type: {eq.getEqType(sf.STD_PRECISION)}")




# print("######")
# print('Symmetric SaddleFocus eq:')
# for eq in tosf.symmetricSaddleFocuses(newEq):
#     print(f"Coords: {eq.coordinates}")



# print("######")
# print('Pairs to check:')
# for eq in tosf.createPairsToCheck(newEq, rhsJac)[0]:
#     print(f"Coords: {eq.coordinates}")


#print(tosf.isSimmetric(newEq[0].coordinates, newEq[1].coordinates))
# pairsToCheck = [newEq]
# info = FH.checkSeparatrixConnection(pairsToCheck, sf.STD_PRECISION, sf.STD_PROXIMITY, rhs,
#                                     jacType, sf.idTransform, sf.pickBothSeparatrices, sf.idListTransform,
#                                     sf.anyNumber, 10, 100., listEqCoords = None)
#
# for i in info:
#     print(i['dist'])