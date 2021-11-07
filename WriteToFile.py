import numpy as np
import systems_fun as sf
import findTHeteroclinic as FH
import time

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


boundsType = [(-0.1, 2*np.pi+0.1), (-0.1, 2*np.pi+0.1)]
bordersType = [(-1e-15, +2 * np.pi + 1e-15), (-1e-15, +2 * np.pi + 1e-15)]

# Gamma = 0.97
# Lambda = 0.2
paramK = 0.06

Gamma = np.linspace(0.89, 0.99, 4)
Lambda = np.linspace(0.15, 0.25, 4)
InfoFile = open('C:/Users/User/my work)/InfoTwoOscillators2.txt', 'w+')
start = time.time()
for paramGamma in Gamma:
    for paramLambda in Lambda:
        InfoFile.write('Gamma = {}, Lambda = {}  '.format(paramGamma, paramLambda))

        Sys = TwoOscillators(paramGamma, paramLambda, paramK)
        TestJacType = Sys.JacType
        TestRhsType = Sys.ReducedSystem
        TestRhs = Sys.FullSystem
        Eq = sf.findEquilibria(TestRhsType, TestJacType, boundsType, bordersType, sf.ShgoEqFinder(300, 30, 1e-10), sf.STD_PRECISION)

        for eq in Eq:
            eq.coordinates = mapBackTo4D(eq.coordinates)

        newEq = []
        for eq in Eq:
            if (sf.is4DSaddleFocusWith1dU(eq, sf.STD_PRECISION)):
                newEq.append(eq)

        pairsToCheck = [newEq]
        info = FH.checkSeparatrixConnection(pairsToCheck, sf.STD_PRECISION, sf.STD_PROXIMITY, TestRhs,
                                            TestJacType, sf.idTransform, sf.pickBothSeparatrices, sf.idListTransform,
                                            sf.anyNumber, 10, 500., listEqCoords = None)
        for i in info:
            InfoFile.write('dist = {}  '.format(i['dist']))
        InfoFile.write('\n')
end = time.time()
print("Took {}s".format(end - start))