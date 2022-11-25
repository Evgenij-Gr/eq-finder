import multiprocessing as mp
import os.path

import numpy as np
import systems_fun as sf
import SystOsscills as a4d
import findTHeteroclinic as fth
import itertools as itls
import time
import plotFun as pf
import scriptUtils as su
import sys
import datetime

from functools import partial

bounds = [(-0.1, +2 * np.pi + 0.1), (-0.1, +2 * np.pi + 0.1)]
bordersEq = [(-1e-15, +2 * np.pi + 1e-15), (-1e-15, +2 * np.pi + 1e-15)]

def workerTresserPairs(params, paramR, pset: sf.PrecisionSettings, eqFinderParams):
    (i, a), (j, b) = params
    r = paramR
    ud = [0.5, a, b, r]
    osc = a4d.FourBiharmonicPhaseOscillators(ud[0], ud[1], ud[2], ud[3])
    nSamp, nIters, zeroToCompare = eqFinderParams
    eqf = sf.ShgoEqFinder(nSamp, nIters, zeroToCompare)
    result = fth.getTresserPairs(osc, bordersEq, bounds, eqf, pset)
    return i, j, a, b, r, result

if __name__ == "__main__":
    if '-h' in sys.argv or '--help' in sys.argv:
        print("Usage: python TresserPairs.py <pathToConfig> <outputMask> <outputDir>"
              "\n    pathToConfig: full path to configuration file (e.g., \"./cfg.txt\")"
              "\n    outputMask: unique name that will be used for saving output"
              "\n    outputDir: directory to which the results are saved")
        sys.exit()
    assert os.path.isfile(sys.argv[1]), "Configuration file does not exist!"
    assert os.path.isdir(sys.argv[3]), "Output directory does not exist!"

    configFile = open("{}".format(sys.argv[1]), 'r')
    configDict = eval(configFile.read())
    eqFinderParams = su.getParamsSHGO(configDict)

    timeOfRun = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')

    N, M, alphas, betas, r = su.getGrid(configDict)

    ps = su.getPrecisionSettings(configDict)

    pool = mp.Pool(mp.cpu_count())
    start = time.time()
    ret = pool.map(partial(workerTresserPairs, paramR = r, pset = ps, eqFinderParams = eqFinderParams ), itls.product(enumerate(alphas), enumerate(betas)))
    end = time.time()
    pool.close()

    nameOutputFile = sys.argv[2]
    pathToOutputDir = sys.argv[3]

    outputFileMask = "{}_{}x{}_{}".format(nameOutputFile, N, M, timeOfRun)
    print("Took {}s".format(end - start))
    pf.savePairsTresserDataAsTxt(pf.prepareTresserPairsData(ret), pathToOutputDir
                             , outputFileMask)