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
bordersEq = [(-1e-9, +2 * np.pi + 1e-9), (-1e-9, +2 * np.pi + 1e-9)]

def workerCheckHeteroclinicSf1Sf2(params, paramR, events, pset: sf.PrecisionSettings, proxs: sf.ProximitySettings, eqFinderParams):
    (i, a), (j, b) = params
    r = paramR
    ud = [0.5, a, b, r]
    osc = a4d.FourBiharmonicPhaseOscillators(ud[0], ud[1], ud[2], ud[3])
    nSamp, nIters, zeroToCompare = eqFinderParams
    eqf = sf.ShgoEqFinder(nSamp, nIters, zeroToCompare)
    result = fth.checkHeterocninicSf1Sf2SaddleLig(osc, bordersEq, bounds, eqf, pset, proxs, 1000., events)
    return i, j, a, b, r, result

if __name__ == "__main__":
    if '-h' in sys.argv or '--help' in sys.argv:
        print("Usage: python HeteroclinicSf1Sf2.py <pathToConfig> <outputMask> <outputDir>"
              "\n    pathToConfig: full path to configuration file (e.g., \"./cfg.txt\")"
              "\n    outputMask: unique name that will be used for saving output"
              "\n    outputDir: directory to which the results are saved")
        sys.exit()
    assert os.path.isfile(sys.argv[1]), "Configuration file does not exist!"
    assert os.path.isdir(sys.argv[3]), "Output directory does not exist!"

    configFile = open("{}".format(sys.argv[1]), 'r')
    configDict = eval(configFile.read())

    timeOfRun = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')

    N, M, alphas, betas, r = su.getGrid(configDict)
    eqFinderParams = su.getParamsSHGO(configDict)

    ps = su.getPrecisionSettings(configDict)
    prox = su.getProximitySettings(configDict)

    evtFlag = configDict['Parameters']['useEvents']

    pool = mp.Pool(mp.cpu_count())
    start = time.time()
    ret = pool.map(partial(workerCheckHeteroclinicSf1Sf2, paramR = r,events = evtFlag, pset = ps, proxs = prox, eqFinderParams = eqFinderParams), itls.product(enumerate(alphas), enumerate(betas)))
    end = time.time()
    pool.close()

    nameOutputFile = sys.argv[2]
    pathToOutputDir = sys.argv[3]
    print("Took {}s".format(end - start))
    outputFileMask = "{}_{}x{}_{}".format(nameOutputFile, N, M, timeOfRun)
    pf.plotHeteroclinicsData(pf.prepareSfsHeteroclinicsData(ret), alphas, betas, r, pathToOutputDir, outputFileMask)
    pf.saveHeteroclinicSfsDataAsTxt(pf.prepareSfsHeteroclinicsData(ret), pathToOutputDir
                                    , outputFileMask)
