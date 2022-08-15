import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import systems_fun as sf
from scipy.integrate import solve_ivp
import os

def saveHeteroclinicsDataAsTxt(HeteroclinicsData, pathToDir, fileName ):
    """
    (i, j, a, b, r, dist, timeIntegration, coordsStartPt, coordsSadfoc, coordsSaddle)
    """
    if HeteroclinicsData:
        headerStr = (
                'i  j  alpha  beta  r  distTrajToEq  integrationTime  startPtX  startPtY  startPtZ  sadfocPtX  sadfocPtY  sadfocPtZ  saddlePtX  saddlePtY  saddlePtZ\n' +
                '0  1  2      3     4  5             6                7         8         9         10         11         12         13         14         15')
        fmtList = ['%2u',
                   '%2u',
                   '%+18.15f',
                   '%+18.15f',
                   '%+18.15f',
                   '%+18.15f',
                   '%+18.15f',
                   '%+18.15f',
                   '%+18.15f',
                   '%+18.15f',
                   '%+18.15f',
                   '%+18.15f',
                   '%+18.15f',
                   '%+18.15f',
                   '%+18.15f',
                   '%+18.15f',
                   ]
        fullOutputName = os.path.join(pathToDir, fileName+'.txt')
        np.savetxt(fullOutputName, HeteroclinicsData, header=headerStr, fmt=fmtList)

def saveHeteroclinicSfsDataAsTxt(HeteroclinicsData, pathToDir, fileName ):
    """
    (i, j, a, b, r, dist, timeIntegration, coordsStartPt, coordsSadfoc, coordsSaddle)
    """
    if HeteroclinicsData:
        headerStr = (
                'i  j  alpha  beta  r  distTrajToEq  integrationTime  startPtX  startPtY  startPtZ  saddle1PtX  saddle1PtY  saddle1PtZ  sadfoc1dUnPtX  sadfoc1dUnPtY  sadfoc1dUnPtZ  sadfoc1dStPtX  sadfoc1dStPtY  sadfoc1dStPtZ\n' +
                '0  1  2      3     4  5             6                7         8         9         10          11          12          13             14             15             16             17             18')
        fmtList = ['%2u',
                   '%2u',
                   '%+18.15f',
                   '%+18.15f',
                   '%+18.15f',
                   '%+18.15f',
                   '%+18.15f',
                   '%+18.15f',
                   '%+18.15f',
                   '%+18.15f',
                   '%+18.15f',
                   '%+18.15f',
                   '%+18.15f',
                   '%+18.15f',
                   '%+18.15f',
                   '%+18.15f',
                   '%+18.15f',
                   '%+18.15f',
                   '%+18.15f',
                   ]
        fullOutputName = os.path.join(pathToDir, fileName+'.txt')
        np.savetxt(fullOutputName, HeteroclinicsData, header=headerStr, fmt=fmtList)

def prepareTargetHeteroclinicsData(data):
    """
        Accepts result of running heteroclinics analysis on grid.
        Expects elements to be tuples in form (i, j, a, b, r, result)
        """
    TargetHeteroclinicsData=[]
    sortedData = sorted(data, key=lambda X: (X[0], X[1]))
    for d in sortedData:
        i, j, alpha, beta, r, infoDicts = d
        if infoDicts:
            for infoDict in infoDicts:
                startPtX, startPtY, startPtZ = infoDict['stPt']
                sadfocPtX, sadfocPtY, sadfocPtZ = infoDict['alpha'].coordinates
                saddlePtX, saddlePtY, saddlePtZ = infoDict['omega'].coordinates
                TargetHeteroclinicsData.append((i, j, alpha, beta, r, infoDict['dist'], infoDict['integrationTime'],
                                                startPtX, startPtY, startPtZ, sadfocPtX, sadfocPtY, sadfocPtZ,
                                                saddlePtX, saddlePtY, saddlePtZ))

    return TargetHeteroclinicsData

def prepareSfsHeteroclinicsData(data):
    """
        Accepts result of running heteroclinics analysis on grid.
        Expects elements to be tuples in form (i, j, a, b, r, result)
        """
    TargetHeteroclinicsData=[]
    sortedData = sorted(data, key=lambda X: (X[0], X[1]))
    for d in sortedData:
        i, j, alpha, beta, r, heterclinInfo = d
        if heterclinInfo:
            for info in heterclinInfo:
                sad, sf1dU, sf1dS, stPt, dist, intTime = info
                startPtX, startPtY, startPtZ = stPt
                saddlePtX, saddlePtY, saddlePtZ = sad.coordinates
                sadfoc1dUnPtX, sadfoc1dUnPtY, sadfoc1dUnPtZ = sf1dU.coordinates
                sadfoc1dSnPtX, sadfoc1dSnPtY, sadfoc1dSnPtZ = sf1dS.coordinates
                TargetHeteroclinicsData.append((i, j, alpha, beta, r, dist, intTime,
                                                startPtX, startPtY, startPtZ, saddlePtX, saddlePtY, saddlePtZ,
                                                sadfoc1dUnPtX, sadfoc1dUnPtY, sadfoc1dUnPtZ,
                                                sadfoc1dSnPtX, sadfoc1dSnPtY, sadfoc1dSnPtZ))

    return TargetHeteroclinicsData

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
    fullOutputName = os.path.join(pathToDir, imageName + '.png')
    plt.savefig(fullOutputName)

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
    fullOutputName = os.path.join(pathToDir, imageName + '.png')
    plt.savefig(fullOutputName)

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
    fullOutputName = os.path.join(pathToDir, imageName + '.png')
    plt.savefig(fullOutputName)

def plotLyapunovMap(LyapunovData, firstParamInterval, secondParamInterval, thirdParamVal, pathToDir,
                    imageName):
    """        (i, j, val)
    """
    N = len(firstParamInterval)
    M = len(secondParamInterval)

    colorGridDist = np.zeros((M, N))
    sortedData = sorted(LyapunovData, key=lambda X: (X[2]))
    for data in sortedData:
        i = int(data[0])
        j = int(data[1])
        if data[2] > 1e-3:
            colorGridDist[j][i] = 1


    plt.pcolormesh(firstParamInterval, secondParamInterval, colorGridDist, cmap=plt.cm.get_cmap('binary'))
    plt.colorbar()
    plt.xlabel(r'$ \alpha $')
    plt.ylabel(r'$ \beta $')
    plt.title("r={}".format(thirdParamVal))
    fullOutputName = os.path.join(pathToDir, imageName + '.png')
    plt.savefig(fullOutputName)

def prepareStartPtsData(data):
    """
        Expects elements to be tuples in form (i, j, a, b, r, result)
        """
    StartPtsData=[]
    sortedData = sorted(data, key=lambda X: (X[0], X[1]))
    for d in sortedData:
        i, j, alpha, beta, r, listStartPt, maxTime, dt = d
        if listStartPt:
            for startPt in listStartPt:
                startPtX, startPtY, startPtZ = startPt
                StartPtsData.append((i, j, alpha, beta, r,  startPtX, startPtY, startPtZ, maxTime, dt))

    return StartPtsData

def saveStartPtsDataAsTxt(prepStartPtsData, pathToDir, fileName):
    if prepStartPtsData:
        headerStr = (
                'i  j  alpha  beta r startPtX  startPtY  startPtZ maxTime dt\n' +
                '0  1  2      3    4 5         6         7        8       9')
        fmtList = ['%2u',
                   '%2u',
                   '%+18.15f',
                   '%+18.15f',
                   '%+18.15f',
                   '%+18.15f',
                   '%+18.15f',
                   '%+18.15f',
                   '%+18.15f',
                   '%+18.15f',]
        fullOutputName = os.path.join(pathToDir, fileName + '.txt')
        np.savetxt(fullOutputName, prepStartPtsData, header=headerStr,
                   fmt=fmtList)

def prepareTresserPairsData(data):
    """
        Expects elements to be tuples in form (i, j, a, b, r, result)
        """
    pairsTresserData=[]
    sortedData = sorted(data, key=lambda X: (X[0], X[1]))
    for d in sortedData:
        i, j, alpha, beta, r, listOfPairsTresser = d
        if listOfPairsTresser:
            for pair in listOfPairsTresser:
                saddle, sadFoc = pair
                saddlePtY, saddlePtZ = saddle.coordinates
                sadfocPtY, sadfocPtZ = sadFoc.coordinates
                pairsTresserData.append((i, j, alpha, beta, r, saddlePtY, saddlePtZ, sadfocPtY, sadfocPtZ))

    return pairsTresserData

def savePairsTresserDataAsTxt(prepStartPtsData, pathToDir, fileName ):
    if prepStartPtsData:
        headerStr = (
                'i  j  alpha  beta r saddlePtY saddlePtZ sadfocPtY sadfocPtZ\n' +
                '0  1  2      3    4 5         6         7         8')
        fmtList = ['%2u',
                   '%2u',
                   '%+18.15f',
                   '%+18.15f',
                   '%+18.15f',
                   '%+18.15f',
                   '%+18.15f',
                   '%+18.15f',
                   '%+18.15f',]

        fullOutputName = os.path.join(pathToDir, fileName + '.txt')
        np.savetxt(fullOutputName, prepStartPtsData, header=headerStr,
                   fmt=fmtList)

def plotHeteroclinicGraph(paramName, titleString, outName, paramLims, heteroclinicParamVal, plotDict):
    pd = {'labelFontsize': 15,
          'titleFontsize': 20,
          'markerSize': 10}
    assert set(plotDict.keys()) <= set(pd.keys()), \
        f"plotDict has wrong keys!\n{list(plotDict.keys()) = }\nvs\n{list(pd.keys()) = }"
    pd.update(plotDict)

    plt.scatter(heteroclinicParamVal, 1, s=pd['markerSize'], c='black')
    plt.xlabel(f"${paramName}$", fontsize=pd['labelFontsize'])
    plt.xlim(paramLims)
    plt.title(titleString)
    plt.tight_layout()
    plt.savefig(outName, facecolor='white')
    plt.close()

def plotHeteroclinicMap(heteroclinicData, xPars, xParName, yPars, yParName, outFileName, titleStr, plotDict):
    """
        approachData is expected to be a list of (i, j, [hetroclinicData])
    """
    pd = {'labelFontsize': 15,
          'colorScheme': 'jet'}
    assert set(plotDict.keys()) <= set(pd.keys()), \
        f"plotDict has wrong keys!\n{list(plotDict.keys()) = }\nvs\n{list(pd.keys()) = }"
    pd.update(plotDict)
    N = len(xPars)
    M = len(yPars)
    timeGrid = np.zeros((M, N))
    for data in heteroclinicData:
        i = data[0]
        j = data[1]
        timeGrid[j][i] = 1

    plt.pcolormesh(xPars,  yPars, timeGrid, cmap=plt.cm.get_cmap(pd['colorScheme']))
    plt.colorbar()
    plt.xlabel(f'${xParName}$', fontsize=pd['labelFontsize'])
    plt.ylabel(f'${yParName}$', fontsize=pd['labelFontsize'])
    plt.title(titleStr)
    plt.tight_layout()
    plt.savefig(outFileName, facecolor='white')
    plt.close()