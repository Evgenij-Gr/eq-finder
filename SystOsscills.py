import numpy as np

class FourBiharmonicPhaseOscillators:
    def __init__(self, paramW, paramA, paramB, paramR):
        self.paramW = paramW
        self.paramA = paramA
        self.paramB = paramB
        self.paramR = paramR

    def funG(self, fi):
        return -np.sin(fi + self.paramA) + self.paramR * np.sin(2 * fi + self.paramB)

    def getFullSystem(self, phis):
        rhsPhis = [0,0,0,0]        
        for i in range(4):
            elem = self.paramW
            for j in range(4):
                elem += 0.25 * self.funG(phis[i]-phis[j])
            rhsPhis[i] = elem
        return rhsPhis

    def getReducedSystem(self, gammas):
        gammas = list(gammas)
        phis = [0] + gammas
        rhsPhi = self.getFullSystem(phis)
        rhsGamma = [0,0,0,0]
        for i in range(4):
            rhsGamma[i] = rhsPhi[i]-rhsPhi[0]
        return rhsGamma[1:]

    def getRestriction(self,psi):       
        psi = list(psi)      
        gammas = [0] + psi
        rhsPsi = self.getReducedSystem(gammas)
        return rhsPsi[1:]
    
    def DiagComponentJac2d(self, x,y):
        
        
        return (2*(-np.cos(x + self.paramA) + 2*self.paramR * np.cos(2 * x + self.paramB)) +
                (-np.cos(x-y + self.paramA) + 2*self.paramR * np.cos(2 * (x-y) + self.paramB)) -
                (np.cos((-x) + self.paramA) - 2*self.paramR * np.cos(2 * (-x) + self.paramB))
               )/4
    def NotDiagComponentJac(self, x,y):
        
        return ((np.cos(x-y + self.paramA) - 2*self.paramR * np.cos(2 * (x-y) + self.paramB)) -
                (np.cos((-y) + self.paramA) - 2*self.paramR * np.cos(2 * (-y) + self.paramB))
               )/4
    
    def DiagComponentJac3d(self, x,y,z):
        
        
        return (
            (-np.cos(x + self.paramA) + 2*self.paramR * np.cos(2 * x + self.paramB)) +
                
            (-np.cos(x-y + self.paramA) + 2*self.paramR * np.cos(2 * (x-y) + self.paramB))+
                
            (-np.cos(x-z + self.paramA) + 2*self.paramR * np.cos(2 * (x-z) + self.paramB))-
                
            (np.cos((-x) + self.paramA) - 2*self.paramR * np.cos(2 * (-x) + self.paramB))
               )/4
        
    def getRestrictionJac(self,X):  
        x,y=X       
        return np.array([[self.DiagComponentJac2d(x,y), self.NotDiagComponentJac(x,y)],
                         [ self.NotDiagComponentJac(y,x),self.DiagComponentJac2d(y,x)]])
    
    def getReducedSystemJac(self,X):  
        x,y,z=X       
        return np.array([[self.DiagComponentJac3d(x,y,z), self.NotDiagComponentJac(x,y),self.NotDiagComponentJac(x,z)],
                         [ self.NotDiagComponentJac(y,x),self.DiagComponentJac3d(y,x,z),self.NotDiagComponentJac(y,z)],
                         [ self.NotDiagComponentJac(z,x),self.NotDiagComponentJac(z,y),self.DiagComponentJac3d(z,x,y)]
                        ])
    
    def getParams(self):          
        return [self.paramW,self.paramA,self.paramB,self.paramR]