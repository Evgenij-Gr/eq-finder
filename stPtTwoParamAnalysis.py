import multiprocessing as mp
import os.path
import numpy as np
import itertools as itls
import time
import plotFun as pf
import scriptUtils as su
import sys
import datetime

from functools import partial


def workerStartPts(params, paramR):
    (i, a), (j, b) = params
    r = paramR
    maxTime = 50000
    evalTs = 0

#    return i, j, a, b, r, [[1.57079632679, 3.14159265359, 4.71238898038 + 0.2]], maxTime, evalTs
    return i, j, a, b, r, [[+1.947080481103596, +2.274942764033958, +4.861822576706475]], maxTime, evalTs
#    return i, j, a, b, r, [[+0.733189690436120, +2.839689764456125, +4.896962002450891]], maxTime, evalTs
#    return i, j, a, b, r, [[1.4696010227307938, 2.396882670704771, 4.139098561506022]], maxTime, evalTs
#    return i, j, a, b, r, [[+2.447080481103596, +2.774942764033958, +5.361822576706475]], maxTime, evalTs

if __name__ == "__main__":
    if '-h' in sys.argv or '--help' in sys.argv:
        print("Usage: python stPtTwoParamAnalysis.py <pathToConfig> <outputMask> <outputDir>"
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

    ps = su.getPrecisionSettings(configDict)

    pool = mp.Pool(mp.cpu_count())
    start = time.time()
    ret = pool.map(partial(workerStartPts, paramR = r), itls.product(enumerate(alphas), enumerate(betas)))
    end = time.time()
    pool.close()

    nameOutputFile = sys.argv[2]
    pathToOutputDir = sys.argv[3]

    outputFileMask = "{}_{}x{}_{}".format(nameOutputFile, N, M, timeOfRun)
    print("Took {}s".format(end - start))
    pf.saveStartPtsDataAsTxt(pf.prepareStartPtsData(ret), pathToOutputDir
                             , outputFileMask)