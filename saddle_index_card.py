import numpy as np
import systems_fun as sf
import TwoPendulumsSystemFun as tpsf
import itertools as itls
from functools import partial
import multiprocessing as mp
from MySystem import TwoPendulums, mapBackTo4D
import matplotlib.pyplot as plt
import matplotlib.colors as colors

ep = sf.EnvironmentParameters('C:/Users/User/eq-finder/output_files/TwoPendulums/Карты седлового индекса', 'eq', 'Image')

boundsType = [(-0.1, 2*np.pi+0.1), (-0.1, 2*np.pi+0.1)]
bordersType = [(-1e-15, +2 * np.pi + 1e-15), (-1e-15, +2 * np.pi + 1e-15)]


def parallSaddle_index_card(params, paramK):
    (i, Gamma), (j, Lambda) = params
    Sys = TwoPendulums(Gamma, Lambda, paramK)
    TestJacType = Sys.JacType
    TestRhsType = Sys.ReducedSystem
    TestRhs = Sys.FullSystem
    JacRhs = Sys.Jac
    Eq = sf.findEquilibria(TestRhs, JacRhs, TestRhsType, TestJacType, mapBackTo4D, boundsType, bordersType,
                           sf.ShgoEqFinder(300, 30, 1e-10), sf.STD_PRECISION)

    newEq = [eq for eq in Eq if sf.is4DSaddleFocusWith1dU(eq, sf.STD_PRECISION)]
    newEq = [eq for eq in newEq if eq.coordinates[0] < eq.coordinates[2]]

    # for eq in newEq:
    #     st = eq.getLeadSEigRe(sf.STD_PRECISION)
    #     print(st)
    #     unst = eq.getLeadUEigRe(sf.STD_PRECISION)
    #     print(unst)
    #     print(f'Седловая величина б = {unst - (-1 * st)}')
    #     print(f'Седловой индекс р = {-st / unst}')
    writeToFileSaddle_index(ep, newEq, [Gamma, Lambda], "{:0>5}_{:0>5}".format(i, j), sf.STD_PRECISION)


def writeToFileSaddle_index(envParams: sf.EnvironmentParameters, EqList, params, nameOfFile, ps: sf.STD_PRECISION):
    sol = []
    for eq in EqList:
        st = eq.getLeadSEigRe(ps)
        # print(st)
        unst = eq.getLeadUEigRe(ps)
        sigma = unst + st
        rho = -st / unst
        # sol.append(st)
        # sol.append(unst)
        # sol.append(sigma)
        isLeadingStable2d = (eq.getEqType(ps)[3] == 1)
        ret = [rho, sigma] if isLeadingStable2d else None
        sol.append(ret)
        # print(ret)
    headerStr = ('gamma = {par[0]}\n' +
                 'lambda = {par[1]}\n' +
                 'SaddleIndex      SaddleValue\n' +
                 '0                1').format(
        par=params)
    if EqList:
        np.savetxt("{env.pathToOutputDirectory}{}.txt".format(nameOfFile, env=envParams), sol, header=headerStr,
               fmt=['%+18.15f',
                   '%+18.15f'])
    else:
        with open("{env.pathToOutputDirectory}{}.txt".format(nameOfFile, env=envParams), "w+") as f:
            f.write("\n")


def createSaddle_index_card(envParams: sf.EnvironmentParameters, numberValuesParam1, numberValuesParam2, arrFirstParam,
                          arrSecondParam):
    M, N = numberValuesParam1, numberValuesParam2
    colorGrid = np.zeros((N, M)) * np.NaN
    for i in range(N):
        for j in range(M):
            try:
                data = np.loadtxt('{}{:0>5}_{:0>5}.txt'.format(envParams.pathToOutputDirectory, i, j), usecols=0)
                colorGrid[i][j] = data
            except BaseException:
                continue

    cmap = (colors.ListedColormap(['silver', 'dimgray']))
    bounds = [0, 1, 3]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    plt.pcolormesh(arrFirstParam, arrSecondParam, colorGrid, cmap=cmap, norm=norm)
    # plt.pcolormesh(arrFirstParam, arrSecondParam, colorGrid, cmap=plt.cm.get_cmap('binary'), norm=colors.CenteredNorm(vcenter=1))
    plt.colorbar()
    plt.xlabel(r'$\lambda$')
    plt.ylabel(r'$\gamma$')
    plt.title('Седловой индекс')
    plt.savefig('{}{}.pdf'.format(envParams.pathToOutputDirectory, envParams.imageStamp))
    plt.clf()


def createSaddle_value_card(envParams: sf.EnvironmentParameters, numberValuesParam1, numberValuesParam2, arrFirstParam,
                          arrSecondParam):
    N, M = numberValuesParam1, numberValuesParam2
    colorGrid = np.zeros((M, N)) * np.NaN
    for i in range(M):
        for j in range(N):
            try:
                data = np.loadtxt('{}{:0>5}_{:0>5}.txt'.format(envParams.pathToOutputDirectory, i, j), usecols=1)
                colorGrid[i][j] = data
            except BaseException:
                continue

    cmap = (colors.ListedColormap(['silver', 'dimgray']))
    bounds = [-5, 0, 5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    plt.pcolormesh(arrFirstParam, arrSecondParam, colorGrid, cmap=cmap, norm=norm)
    # plt.pcolormesh(arrFirstParam, arrSecondParam, colorGrid, cmap=plt.cm.get_cmap('binary'), norm=colors.CenteredNorm(vcenter=1))
    plt.colorbar()
    plt.xlabel('Gamma')
    plt.ylabel('Lambda')
    plt.title('Седловая величина')
    plt.savefig('{}{}.pdf'.format(envParams.pathToOutputDirectory, envParams.imageStamp))
    plt.clf()


if __name__ == "__main__":
    configFile = open('C:/Users/User/eq-finder/config.txt', 'r')
    configDict = eval(configFile.read())
    N, M, gammas, lambdas, paramK = tpsf.get_grid(configDict)
    pool = mp.Pool(mp.cpu_count())
    pool.map(partial(parallSaddle_index_card, paramK=paramK),
                   itls.product(enumerate(gammas), enumerate(lambdas)))
    pool.close()

    createSaddle_index_card(ep, M, N, lambdas, gammas)