import numpy as np
import systems_fun as sf
import os
import matplotlib.pyplot as plt


def distanceOnCircle(fi1, fi2):
    dist = min(abs(fi1-fi2), abs(fi1-fi2+2*np.pi))
    return dist


def periodDistance4D(point1, point2):
    fi11, V11, fi21, V21 = point1
    fi12, V12, fi22, V22 = point2
    dist = np.sqrt((distanceOnCircle(fi11, fi12))**2 + (V11-V12)**2 +
                   (distanceOnCircle(fi21, fi22))**2 + (V21-V22)**2)
    return dist


def Distance4D(separatrix, coordinates):
    arrayDistance = []
    for point in separatrix:
        dist = periodDistance4D(point, coordinates)
        arrayDistance.append(dist)
    return min(arrayDistance)


# septest = [[4, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0]]
coordinatestest1 = [1, 2, 3, 4]
coordinatestest2 = [3, 4, 1, 2]
coordinatestest3 = [4, 3, 2, 1]
# print(Distance4D(septest, coordinatestest))


def T(eqCoords):
    fi1, V1, fi2, V2 = eqCoords
    return [fi2, V2, fi1, V1]


# def isSymmetric(eq1, eq2):
#     return (eq1 == T(eq2))

def symmetricSaddleFocuses(Eq):
    symmetricEq = [eq for eq in Eq if eq.coordinates[0] < eq.coordinates[2]]
    return symmetricEq


def applyTtoEq(eq, rhsJac):
    cds = eq.coordinates
    newCds = T(cds)
    return sf.getEquilibriumInfo(newCds, rhsJac)


def createPairsToCheck(newEq, rhsJac):
    filtSadFocList = symmetricSaddleFocuses(newEq)
    pairsToCheck = [(eq, applyTtoEq(eq, rhsJac)) for eq in filtSadFocList]
    return pairsToCheck

# print(T(coordinatestest1))
# print(isSimmetric(coordinatestest1, coordinatestest2))
# print(isSimmetric(coordinatestest1, coordinatestest3))

def prepareTwoOscHeteroclinicsData(data):
    #(i, j, gamma, lambda, k, result)
    HeteroclinicsData = []
    sortedData = sorted(data, key=lambda X: (X[0], X[1]))
    for d in sortedData:
        i, j, Gamma, Lambda, paramK, infoDicts = d
        if infoDicts:
            for infoDict in infoDicts:
                startPtfi1, startPtV1, startPtfi2, startPtV2 = infoDict['stPt']
                sadfocPtfi1, sadfocPtV1, sadfocPtfi2, sadfocPtV2 = infoDict['alpha'].coordinates
                saddlePtfi1, saddlePtV1, saddlePtfi2, saddlePtV2 = infoDict['omega'].coordinates
                HeteroclinicsData.append((i, j, Gamma, Lambda, paramK, infoDict['dist'], infoDict['integrationTime'],
                                          startPtfi1, startPtV1, startPtfi2, startPtV2,
                                          sadfocPtfi1, sadfocPtV1, sadfocPtfi2, sadfocPtV2,
                                          saddlePtfi1, saddlePtV1, saddlePtfi2, saddlePtV2))

    return HeteroclinicsData


def saveTwoOscHeteroclinicsDataAsTxt(HeteroclinicsData, pathToDir, fileName):
    """
    (i, j, a, b, r, dist, timeIntegration, coordsStartPt, coordsSadfoc, coordsSaddle)
    """
    if HeteroclinicsData:
        headerStr = (
                'i   j  Gamma              Lambda             paramK             distTrajToEq       integrationTime      startPtfi1         startPtV1          startPtfi2         startPtV2          sadfocPtfi1        sadfocPtV1         sadfocPtfi2        sadfocPtV2         saddlePtfi1        saddlePtV1         saddlePtfi2        saddlePtV2\n' +
                '0   1  2                  3                  4                  5                  6                    7                  8                  9                  10                 11                 12                 13                 14                 15                 16                 17                 18')
        fmtList = ['%3u',
                   '%3u',
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
        fullOutputName = os.path.join(pathToDir, fileName + '.txt')
        np.savetxt(fullOutputName, HeteroclinicsData, header=headerStr, fmt=fmtList)


def plotTwoOscHeteroclinicsData(heteroclinicsData, firstParamInterval, secondParamInterval, thirdParamVal, pathToDir, imageName):
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
    plt.xlabel(r'$ \Gamma $')
    plt.ylabel(r'$ \Lambda $')
    plt.title("K={}".format(thirdParamVal))
    fullOutputName = os.path.join(pathToDir, imageName + '.png')
    plt.savefig(fullOutputName)