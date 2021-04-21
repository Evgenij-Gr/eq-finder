import multiprocessing as mp
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

def workerCheckTarget(params, paramR, events, pset: sf.PrecisionSettings, proxs: sf.ProximitySettings):
    (i, a), (j, b) = params
    ud = [0.5, a, b, paramR]
    osc = a4d.FourBiharmonicPhaseOscillators(ud[0], ud[1], ud[2], ud[3])
    eqf = sf.ShgoEqFinder(300, 30, 1e-10)
    result = fth.checkTargetHeteroclinic(osc, bordersEq, bounds, eqf, pset, proxs, 1000., events)
    return i, j, a, b, result

if __name__ == "__main__":
    f = open("{}".format(sys.argv[1]), 'r')
    d = eval(f.read())

    timeOfRun = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')

    N, M, alphas, betas, r = su.getGrid(d)

    ps = su.getPrecisionSettings(d)
    prox = su.getProximitySettings(d)

    evnts = d['Parameters']['withEvents']

    pool = mp.Pool(mp.cpu_count())
    start = time.time()
    ret = pool.map(partial(workerCheckTarget, paramR = r, events = evnts, pset = ps, proxs = prox, ), itls.product(enumerate(alphas), enumerate(betas)))
    end = time.time()
    pool.close()

    nameOutputFile = sys.argv[2]
    pathToOutputDir = sys.argv[3]

    print("Took {}s".format(end - start))

    pf.plotHeteroclinicsData(pf.prepareHeteroclinicsData(ret, r), alphas, betas, r, pathToOutputDir
                             , "{}_{}x{}_{}".format(nameOutputFile, N, M, timeOfRun))
    pf.saveHeteroclinicsDataAsTxt(pf.prepareHeteroclinicsData(ret, r), pathToOutputDir
                                  , "{}_{}x{}_{}".format(nameOutputFile, N, M, timeOfRun))
