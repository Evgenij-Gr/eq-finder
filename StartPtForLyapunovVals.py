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
from functools import partial

bounds = [(-0.1, +2 * np.pi + 0.1), (-0.1, +2 * np.pi + 0.1)]
bordersEq = [(-1e-15, +2 * np.pi + 1e-15), (-1e-15, +2 * np.pi + 1e-15)]

def workerStartPts(params, pset: sf.PrecisionSettings):
    (i, a), (j, b) = params
    ud = [0.5, a, b, 1]
    osc = a4d.FourBiharmonicPhaseOscillators(ud[0], ud[1], ud[2], ud[3])
    eqf = sf.ShgoEqFinder(300, 30, 1e-10)
    result = fth.getStartPtsForLyapVals(osc, bordersEq, bounds, eqf, pset, OnlySadFoci = False)
    return i, j, a, b, result

if __name__ == "__main__":
    f = open("{}{}.txt".format(sys.argv[1], sys.argv[2]), 'r')
    d = eval(f.read())
    N, M, alphas, betas, r = su.getGrid(d)
    ps = su.getPrecisionSettings(d)
    pool = mp.Pool(mp.cpu_count())
    start = time.time()
    ret = pool.map(partial(workerStartPts, pset = ps), itls.product(enumerate(alphas), enumerate(betas)))
    end = time.time()
    pool.close()
    pf.saveStartPtsDataAsTxt(pf.prepareStartPtsData(ret, r), './output_files/LyapVals/'
                             , "StartPts{}x{}".format(N, M))