import systems_fun as sf

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
    prox = sf.ProximitySettings(toSinkPrxtyEv = dictConfig['ConnectionProximity']['toSinkPrxtyEv'],
                                toSddlPrxtyEv = dictConfig['ConnectionProximity']['toSddlPrxtyEv'],
                                toTargetSinkPrxtyEv=dictConfig['ConnectionProximity']['toTargetSinkPrxtyEv'],
                                toTargetSddlPrxtyEv=dictConfig['ConnectionProximity']['toTargetSddlPrxtyEv'],
                                toSinkPrxty = dictConfig['ConnectionProximity']['toSinkPrxty'],
                                toSddlPrxty = dictConfig['ConnectionProximity']['toSddlPrxty'])
    return prox