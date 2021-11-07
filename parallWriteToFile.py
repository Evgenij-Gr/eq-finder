import numpy as np

import TwoOscillatorsSystemFun as tosf
import systems_fun as sf
import findTHeteroclinic as FH
import multiprocessing as mp
import itertools as itls
import time
from functools import partial
from MySystem import TwoOscillators, mapBackTo4D


boundsType = [(-0.1, 2*np.pi+0.1), (-0.1, 2*np.pi+0.1)]
bordersType = [(-1e-15, +2 * np.pi + 1e-15), (-1e-15, +2 * np.pi + 1e-15)]

paramK = 0.06

Gamma = np.linspace(0.6, 0.8, 3)
Lambda = np.linspace(0.15, 0.35, 3)


def parallWrite(params, paramK):
    (i, Gamma), (j, Lambda) = params
    Sys = TwoOscillators(Gamma, Lambda, paramK)
    TestJacType = Sys.JacType
    TestRhsType = Sys.ReducedSystem
    TestRhs = Sys.FullSystem
    JacRhs = Sys.Jac
    Eq = sf.findEquilibria(TestRhsType, TestJacType, boundsType, bordersType, sf.ShgoEqFinder(300, 30, 1e-10), sf.STD_PRECISION)

    for eq in Eq:
        eq.coordinates = mapBackTo4D(eq.coordinates)

    newEq = [eq for eq in Eq if sf.is4DSaddleFocusWith1dU(eq, sf.STD_PRECISION)]

    pairsToCheck = tosf.createPairsToCheck(newEq, JacRhs)
    info = FH.checkSeparatrixConnection(pairsToCheck, sf.STD_PRECISION, sf.STD_PROXIMITY, TestRhs,
                                            TestJacType, sf.idTransform, sf.pickBothSeparatrices, sf.idListTransform,
                                            sf.anyNumber, 1.5, 500., 1,  listEqCoords = None)
    return i, j, Gamma, Lambda, paramK, info


if __name__ == "__main__":
    pool = mp.Pool(mp.cpu_count())
    start = time.time()
    ret = pool.map(partial(parallWrite, paramK = paramK), itls.product(enumerate(Gamma), enumerate(Lambda)))
    end = time.time()
    pool.close()
    print("Took {}s".format(end - start))
    preparedData = tosf.prepareTwoOscHeteroclinicsData(ret)
    tosf.saveTwoOscHeteroclinicsDataAsTxt(preparedData, 'C:/Users/User/eq-finder)', 'TwoOscillatorsHeteroclinicsData4')
    tosf.plotTwoOscHeteroclinicsData(preparedData, Gamma, Lambda, paramK, 'C:/Users/User/eq-finder', 'TwoOscillatorsHeteroclinicsData4')