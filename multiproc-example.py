import multiprocessing as mp
import numpy as np
import systems_fun as sf
import SystOsscills as sys
import findTHeteroclinic as fth
import itertools as itls
import time

# Задаем область поиска локальных минимумов для численного метода, а также более точную область поиска.

bounds = [(-0.1, +2 * np.pi + 0.1), (-0.1, +2 * np.pi + 0.1)]
bordersEq = [(-1e-15, +2 * np.pi + 1e-15), (-1e-15, +2 * np.pi + 1e-15)]

# Разбиваем значения параметров на сетку

N = 11  # Количество разбиений параметра альфа
M = 11  # Количество разбиений параметра бета

alphas = np.linspace(0, 2 * np.pi, N)
betas = np.linspace(0, 2 * np.pi, M)

ps = sf.STD_PRECISION

def workerChectTargHet(params):
    (i, a), (j, b) = params
    ud = [0.5, a, b, 1]
    osc = sys.FourBiharmonicPhaseOscillators(ud[0], ud[1], ud[2], ud[3])
    eqf = sf.ShgoEqFinder(300, 30, 1e-10)
    result = fth.checkTargetHeteroclinic(osc, bordersEq, bounds, eqf, sf.STD_PRECISION, sf.STD_PROXIMITY,1000. )
    return i, j, a, b, result

if __name__ == "__main__":
    pool = mp.Pool(mp.cpu_count())
    start = time.time()
    ret = pool.map(workerChectTargHet, itls.product(enumerate(alphas), enumerate(betas)))
    end = time.time()
    pool.close()
    print("Took {}s".format(end - start))
    print(ret)
