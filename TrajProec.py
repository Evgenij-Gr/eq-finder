import multiprocessing as mp
import numpy as np
import systems_fun as sf
import SystOsscills as a4d
import itertools as itls
import time
import plotFun as pf
import sys
import scriptUtils as su
from functools import partial

def workerPlotTrajProec(params, pset: sf.PrecisionSettings, dictConfig):
    (i, (a, b)), startPt = params
    ud = [0.5, a, b, 1]
    osc = a4d.FourBiharmonicPhaseOscillators(ud[0], ud[1], ud[2], ud[3])
    pf.plotTrajProec(osc.getReducedSystem, startPt, pset, 1000, './output_files/TrajProec/', "TrajProec_{}_{}".format(i,dictConfig['InputFile']['NameOfFile']),a,b)

if __name__ == "__main__":
    f = open("{}{}.txt".format(sys.argv[1], sys.argv[2]), 'r')
    d = eval(f.read())
    DtFile = "{}{}.txt".format(d['InputFile']['PathToFile'], d['InputFile']['NameOfFile'])
    pars = np.loadtxt(DtFile, usecols=(2, 3))
    startPts = np.loadtxt(DtFile, usecols=(5, 6, 7))
    pool = mp.Pool(mp.cpu_count())
    ps = su.getPrecisionSettings(d)
    start = time.time()
    ret = pool.map(partial(workerPlotTrajProec, pset = ps, di = d), itls.product(enumerate(pars), startPts))
    end = time.time()
    pool.close()