import numpy as np


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