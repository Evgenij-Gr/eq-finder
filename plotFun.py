import numpy as np
import matplotlib.pyplot as plt
import systems_fun as sf
import matplotlib
from scipy.integrate import solve_ivp

def saveHeteroclinicsDataAsTxt(HeteroclinicsData, pathToDir, fileName ):
    """
    (i, j, a, b, r, dist)
    """
    if HeteroclinicsData:
        headerStr = (
                'i  j  alpha  beta  r    dist startPtX  startPtY  startPtZ\n' +
                '0  1  2      3     4    5    6         7         8')
        fmtList = ['%2u',
                   '%2u',
                   '%+18.15f',
                   '%+18.15f',
                   '%+18.15f',
                   '%+18.15f',
                   '%+18.15f',
                   '%+18.15f',
                   '%+18.15f',]
        np.savetxt("{}{}.txt".format(pathToDir,fileName), HeteroclinicsData, header=headerStr,
                   fmt=fmtList)

def prepareHeteroclinicsData(data, r):
    """
        Accepts result of running heteroclinics analysis on grid.
        Expects elements to be tuples in form (i, j, a, b, result)
        """
    HeteroclinicsData=[]
    sortedData = sorted(data, key=lambda X: (X[0], X[1]))
    for d in sortedData:
        if d[4]:
            for dt in d[4]:
            #dict = min(d[4], key=lambda i: i['dist'])
                Xpt,Ypt,Zpt = dt['stPt']
                HeteroclinicsData.append((d[0], d[1], d[2], d[3], r, dt['dist'], Xpt, Ypt, Zpt))

    return HeteroclinicsData

def plotHeteroclinicsData(heteroclinicsData, firstParamInterval ,secondParamInterval, thirdParamVal, pathToDir, imageName):
    """
    (i, j, a, b, r, dist)
    """
    N = len(firstParamInterval)
    M = len(secondParamInterval)

    colorGridDist = np.zeros((M, N))

    for data in heteroclinicsData:
        i = data[0]
        j = data[1]
        colorGridDist[j][i] = 1

    plt.pcolormesh(firstParamInterval, secondParamInterval, colorGridDist, cmap=plt.cm.get_cmap('binary'))
    plt.colorbar()
    plt.xlabel(r'$ \alpha $')
    plt.ylabel(r'$ \beta $')
    plt.title("r={}".format(thirdParamVal))
    plt.savefig("{}{}".format(pathToDir,imageName))

def plotTresserPairs(osc, bounds, bordersEq, ps, pathToDir, imageName):
    eqList = sf.findEquilibria(osc.getRestriction, osc.getRestrictionJac, bounds, bordersEq,
                               sf.ShgoEqFinder(300, 30, 1e-10),ps)
    gfe = sf.getTresserPairs(eqList, osc, ps)

    xs = ys = np.linspace(0, +2 * np.pi, 1001)
    res = np.zeros([len(xs), len(xs)])
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            res[i][j] = np.log10(np.dot(osc.getRestriction([x, y]), osc.getRestriction([x, y])) + 1e-10)

    matplotlib.rcParams['figure.figsize'] = 10, 10

    plt.pcolormesh(xs, ys, res, cmap=plt.cm.get_cmap('RdBu'))
    plt.xlim([0, +2 * np.pi])
    plt.ylim([0, +2 * np.pi])
    plt.xlabel('$\gamma_3$')
    plt.ylabel('$\gamma_4$')
    plt.axes().set_aspect('equal', adjustable='box')
    for pair in gfe:
        saddle, sadfoc = pair
        p1 = plt.scatter(saddle.coordinates[0], saddle.coordinates[1], c='green', s=40)
        p2 = plt.scatter(sadfoc.coordinates[0], sadfoc.coordinates[1], c='red', s=40)
    plt.legend([p1, p2], ["Седло", "Седло-фокус"])
    plt.savefig("{}{}".format(pathToDir, imageName))

def plotTrajProec(osc, startPt, ps, maxTime, pathToDir, imageName, a, b):
    rhs_vec = lambda t, X: osc(X)
    sep = solve_ivp(rhs_vec, [0, maxTime], startPt, rtol=ps.rTol, atol=ps.aTol, dense_output=True)

    x = sep.y[0]
    y = sep.y[1]
    z = sep.y[2]

    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(30, 10))

    traj =(x,y,z)
    ParamsTraj = [(0,1,r'$\phi_1$',r'$\phi_2$'),(0,2,r'$\phi_1$',r'$\phi_3$'),(1,2,r'$\phi_2$',r'$\phi_3$')]

    for i,params in enumerate(ParamsTraj):
        IndxFirstCoord, IndxSecondCoord, firstLab, secondLab = params
        axs[i].scatter(traj[IndxFirstCoord][0], traj[IndxSecondCoord][0], s=40, c='g', label='Начало')
        axs[i].scatter(traj[IndxFirstCoord][-1], traj[IndxSecondCoord][-1], s=40, c='r', label='Конец')
        axs[i].set_xlim(0, 2 * np.pi)
        axs[i].set_ylim(0, 2 * np.pi)
        axs[i].plot(traj[IndxFirstCoord], traj[IndxSecondCoord])
        axs[i].set_xlabel(firstLab)
        axs[i].set_ylabel(secondLab)

    axs[0].set_title(r'$\alpha ={}, \beta ={}$'.format(a,b))
    axs[0].legend()
    plt.savefig("{}{}".format(pathToDir,imageName))

def plotLyapunovMap(LyapunovData, firstParamInterval, secondParamInterval, thirdParamVal, pathToDir,
                    imageName):
    """        (i, j, val)
    """
    N = len(firstParamInterval)
    M = len(secondParamInterval)

    colorGridDist = np.zeros((M, N))
    sortedData = sorted(LyapunovData, key=lambda X: (X[2]), reverse=True)
    for data in sortedData:
        i = int(data[0])
        j = int(data[1])
        if data[2] > 1e-2:
            colorGridDist[j][i] = 1



    plt.pcolormesh(firstParamInterval, secondParamInterval, colorGridDist, cmap=plt.cm.get_cmap('binary'))
    plt.colorbar()
    plt.xlabel(r'$ \alpha $')
    plt.ylabel(r'$ \beta $')
    plt.title("r={}".format(thirdParamVal))
    plt.savefig("{}{}".format(pathToDir, imageName))

def prepareStartPtsData(data, paramR):
    """
        Expects elements to be tuples in form (i, j, a, b, result) and float r
        """
    StartPtsData=[]
    sortedData = sorted(data, key=lambda X: (X[0], X[1]))
    for tup in sortedData:
        if tup[4]:
            for xyz in tup[4]:
                Xpt,Ypt,Zpt = xyz
                StartPtsData.append((tup[0], tup[1], tup[2], tup[3], paramR, Xpt,Ypt,Zpt))

    return StartPtsData

def saveStartPtsDataAsTxt(prepStartPtsData, pathToDir, fileName ):
    if prepStartPtsData:
        headerStr = (
                'i  j  alpha  beta r startPtX  startPtY  startPtZ\n' +
                '0  1  2      3    4 5         6         7')
        fmtList = ['%2u',
                   '%2u',
                   '%+18.15f',
                   '%+18.15f',
                   '%+18.15f',
                   '%+18.15f',
                   '%+18.15f',
                   '%+18.15f',]
        np.savetxt("{}{}.txt".format(pathToDir,fileName), prepStartPtsData, header=headerStr,
                   fmt=fmtList)