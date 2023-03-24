import numpy as np
import sys
import TwoPendulumsSystemFun as tpsf
import systems_fun as sf
import findTHeteroclinic as FH
import multiprocessing as mp
import itertools as itls
import time
from functools import partial
from MySystem import TwoPendulums, mapBackTo4D
import datetime
import os.path

boundsType = [(-0.1, 2*np.pi+0.1), (-0.1, 2*np.pi+0.1)]
bordersType = [(-1e-15, +2 * np.pi + 1e-15), (-1e-15, +2 * np.pi + 1e-15)]


def parallWrite(params, paramK, events=False):
    (i, Gamma), (j, Lambda) = params

    Sys = TwoPendulums(Gamma, Lambda, paramK)
    TestJacType = Sys.JacType
    TestRhsType = Sys.ReducedSystem
    TestRhs = Sys.FullSystem
    JacRhs = Sys.Jac
    Eq = sf.findEquilibria(TestRhs, JacRhs, TestRhsType, TestJacType, mapBackTo4D, boundsType, bordersType, sf.ShgoEqFinder(300, 30, 1e-10), sf.STD_PRECISION)

    # for eq in Eq:
    #     eq.coordinates = mapBackTo4D(eq.coordinates)

    newEq = [eq for eq in Eq if sf.is4DSaddleFocusWith1dU(eq, sf.STD_PRECISION)]

    if events:
        allSymmEqs = itls.chain.from_iterable([[eq, tpsf.applyTtoEq(eq, JacRhs)] for eq in Eq])
    else:
        allSymmEqs = None

    # pairs_to_check = tpsf.createPairsToCheck(newEq, JacRhs)
    # cnctInfo = FH.checkSeparatrixConnection(pairs_to_check, sf.STD_PRECISION, sf.STD_PROXIMITY, TestRhs,
    #                                         TestJacType, sf.idTransform, sf.pickBothSeparatrices, sf.idListTransform,
    #                                         sf.anyNumber, 0.0063, 1000., tpsf.periodDistance4D, listEqCoords=allSymmEqs)
    pairs_to_check = [[newEq[0], newEq[0]]]
    cnctInfo = FH.checkSeparatrixConnection(pairs_to_check, sf.STD_PRECISION, sf.STD_PROXIMITY, TestRhs,
                                            TestJacType, sf.idTransform, sf.pickBothSeparatrices, sf.idListTransform,
                                            sf.anyNumber, 1, 1000., tpsf.periodDistance4D, listEqCoords=[newEq[0]])

    print('{}-{}'.format(i, j))

    return i, j, Gamma, Lambda, paramK, cnctInfo




if __name__ == "__main__":
    if '-h' in sys.argv or '--help' in sys.argv:
        print("Usage: python TargetHeteroclinic.py <pathToConfig> <outputMask> <outputDir>"
              "\n    pathToConfig: full path to configuration file (e.g., \"./cfg.txt\")"
              "\n    outputMask: unique name that will be used for saving output"
              "\n    outputDir: directory to which the results are saved")
        sys.exit()
    assert os.path.isfile(sys.argv[1]), "Configuration file does not exist!"
    assert os.path.isdir(sys.argv[3]), "Output directory does not exist!"

    configFile = open("{}".format(sys.argv[1]), 'r')
    configDict = eval(configFile.read())

    timeOfRun = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')

    N, M, gammas, lambdas, paramK = tpsf.get_grid(configDict)

    evtFlag = configDict['Parameters']['useEvents']

    pool = mp.Pool(mp.cpu_count())
    start = time.time()
    ret = pool.map(partial(parallWrite, paramK=paramK, events=evtFlag), itls.product(enumerate(gammas), enumerate(lambdas)))
    end = time.time()
    pool.close()

    nameOutputFile = sys.argv[2]
    pathToOutputDir = sys.argv[3]
    print("Took {}s".format(end - start))
    outputFileMask = "{}_{}x{}_{}".format(nameOutputFile, N, M, timeOfRun)

    preparedData = tpsf.prepareTwoPendulumsHeteroclinicsData(ret)
    tpsf.saveTwoPendulumsHeteroclinicsDataAsTxt(preparedData, pathToOutputDir, outputFileMask)
    tpsf.plotTwoPendulumsHeteroclinicsData(preparedData, gammas, lambdas, paramK, pathToOutputDir, outputFileMask)

# TODO python parallWriteToFile.py C:\Users\User\eq-finder\config.txt TwoPendulumsHeteroclinicsData C:\Users\User\eq-finder\output_files\TwoPendulums\Гомоклиника_карты_расстояний\k=0_06
