import numpy as np
import systems_fun as sf

def getGrid(dictConfig):
    N = dictConfig['Parameters']['a_N']  # Количество разбиений параметра альфа
    M = dictConfig['Parameters']['b_N']  # Количество разбиений параметра бета

    alphas = np.linspace(dictConfig['Parameters']['a_min'], dictConfig['Parameters']['a_max'], N)
    betas = np.linspace(dictConfig['Parameters']['b_min'], dictConfig['Parameters']['b_max'], M)

    r = dictConfig['Parameters']['rval']
    return ( N, M, alphas, betas, r)

def getPrecisionSettings(dictConfig):
    ps = sf.PrecisionSettings(zeroImagPartEps=dictConfig['NumericTolerance']['zeroImagPartEps'],
                              zeroRealPartEps=dictConfig['NumericTolerance']['zeroRealPartEps'],
                              clustDistThreshold=dictConfig['NumericTolerance']['clustDistThreshold'],
                              separatrixShift=dictConfig['SeparatrixComputing']['separatrixShift'],
                              separatrix_rTol=dictConfig['SeparatrixComputing']['separatrix_rTol'],
                              separatrix_aTol=dictConfig['SeparatrixComputing']['separatrix_aTol'],
                              marginBorder=dictConfig['NumericTolerance']['marginBorder']
                              )
    return ps

def getProximitySettings(dictConfig):
    prox = sf.ProximitySettings(toSinkPrxty = dictConfig['ConnectionProximity']['toSinkPrxty'],
                                toSddlPrxty = dictConfig['ConnectionProximity']['toSddlPrxty'])
    return prox