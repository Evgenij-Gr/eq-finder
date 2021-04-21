# example of processing a grid
# in parameter space

import systems_fun as sf
import SystOsscills as a4d
import numpy as np
import findTHeteroclinic as fth
import time

#Задаем область поиска локальных минимумов для численного метода, а также более точную область поиска.

bounds = [(-0.1,+2*np.pi+0.1), (-0.1,+2*np.pi+0.1)]
bordersEq = [(-1e-15,+2*np.pi+1e-15), (-1e-15,+2*np.pi+1e-15)]

# Разбиваем значения параметров на сетку

N = 7 # Количество разбиений параметра альфа
M = 7 # Количество разбиений параметра бета

alphas = np.linspace(0, 2 * np.pi, N)
betas = np.linspace(0, 2*np.pi, M)

ps = sf.STD_PRECISION

start = time.time()

for i,a in enumerate(alphas):
    for j,b in enumerate(betas):
        # Устанавливаем значения параметров системы
        ud = [0.5,a,b,1]
        osc = a4d.FourBiharmonicPhaseOscillators(ud[0], ud[1], ud[2], ud[3])
        eqf = sf.ShgoEqFinder(300, 30, 1e-10)
        ret = fth.checkTargetHeteroclinic(osc, bordersEq, bounds, eqf, sf.STD_PRECISION, sf.STD_PROXIMITY, 1000.)

end = time.time()
print("Took {}s".format(end-start))