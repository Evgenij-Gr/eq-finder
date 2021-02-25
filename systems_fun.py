import os
import scipy
import matplotlib.pyplot as plt
import numpy as np

from scipy import optimize
from numpy import linalg as LA
from sklearn.cluster import AgglomerativeClustering
from scipy.integrate import solve_ivp


class EnvironmentParameters:
    def __init__(self, pathToOutputDirectory, outputStamp, imageStamp):
        assert (os.path.isdir(pathToOutputDirectory)), 'Output directory does not exist!'
        self.pathToOutputDirectory = os.path.join(os.path.normpath(pathToOutputDirectory), '')
        self.outputStamp = outputStamp
        self.imageStamp = imageStamp


class PrecisionSettings:
    def __init__(self, zeroImagPartEps, zeroRealPartEps, clustDistThreshold, separatrixShift, separatrix_rTol,
                 separatrix_aTol):
        assert zeroImagPartEps > 0, "Precision must be greater than zero!"
        assert zeroRealPartEps > 0, "Precision must be greater than zero!"
        assert clustDistThreshold > 0, "Precision must be greater than zero!"
        self.zeroImagPartEps = zeroImagPartEps
        self.zeroRealPartEps = zeroRealPartEps
        self.clustDistThreshold = clustDistThreshold
        self.separatrixShift = separatrixShift
        self.rTol = separatrix_rTol
        self.aTol = separatrix_aTol

    def isEigStable(self, Z):
        return Z.real < -self.zeroRealPartEps

    def isEigUnstable(self, Z):
        return Z.real > +self.zeroRealPartEps

    def isComplex(self, Z):
        return abs(Z.imag) > self.zeroImagPartEps


STD_PRECISION = PrecisionSettings(zeroImagPartEps=1e-14,
                                  zeroRealPartEps=1e-14,
                                  clustDistThreshold=1e-5,
                                  separatrixShift=1e-5,
                                  separatrix_rTol=1e-11,
                                  separatrix_aTol=1e-11)


class Equilibrium:
    def __init__(self, coordinates, eigenvalues, eigvectors):
        eigPairs = list(zip(eigenvalues, eigvectors))
        eigPairs = sorted(eigPairs, key=lambda p: p[0].real)
        eigvalsNew, eigvectsNew = zip(*eigPairs)
        self.coordinates = list(coordinates)
        self.eigenvalues = eigvalsNew
        self.eigvectors = eigvectsNew
        if len(eigenvalues) != len(coordinates):
            raise ValueError('Vector of coordinates and vector of eigenvalues must have the same size!')

    def strToFile(self, ps: PrecisionSettings):
        eigs = []
        for val in self.eigenvalues:
            eigs += [val.real, val.imag]
        return self.coordinates + self.getEqType(ps) + eigs

    def getLeadSEigRe(self, ps: PrecisionSettings):
        return max([se.real for se in self.eigenvalues if ps.isEigStable(se)])

    def getLeadUEigRe(self, ps: PrecisionSettings):
        return min([se.real for se in self.eigenvalues if ps.isEigUnstable(se)])

    def getEqType(self, ps: PrecisionSettings):
        return describeEqType(np.array(self.eigenvalues), ps)


def describeEqType(eigvals, ps: PrecisionSettings):
    eigvalsS = eigvals[np.real(eigvals) < -ps.zeroRealPartEps]
    eigvalsU = eigvals[np.real(eigvals) > +ps.zeroRealPartEps]
    nS = len(eigvalsS)
    nU = len(eigvalsU)
    nC = len(eigvals) - nS - nU
    issc = 1 if nS > 0 and ps.isComplex(eigvalsS[-1]) else 0
    isuc = 1 if nU > 0 and ps.isComplex(eigvalsU[0]) else 0
    return [nS, nC, nU, issc, isuc]


def describePortrType(arrEqSignatures):
    phSpaceDim = int(sum(arrEqSignatures[0]))
    eqTypes = {(i, phSpaceDim - i): 0 for i in range(phSpaceDim + 1)}
    nonRough = 0
    for eqSign in arrEqSignatures:
        nS, nC, nU = eqSign
        if nC == 0:
            eqTypes[(nU, nS)] += 1
        else:
            nonRough += 1
    # nSinks, nSaddles, nSources,  nNonRough
    portrType = tuple([eqTypes[(i, phSpaceDim - i)] for i in range(phSpaceDim + 1)] + [nonRough])
    return portrType


class ShgoEqFinder:
    def __init__(self, nSamples, nIters, eps):
        self.nSamples = nSamples
        self.nIters = nIters
        self.eps = eps

    def __call__(self, rhs, rhsSq, rhsJac, boundaries, borders):
        optResult = scipy.optimize.shgo(rhsSq, boundaries, n=self.nSamples, iters=self.nIters, sampling_method='sobol');
        allEquilibria = [x for x, val in zip(optResult.xl, optResult.funl) if
                         abs(val) < self.eps and inBounds(x, borders)];
        return allEquilibria


class NewtonEqFinder:
    def __init__(self, xGridSize, yGridSize, eps):
        self.xGridSize = xGridSize
        self.yGridSize = yGridSize
        self.eps = eps

    def __call__(self, rhs, rhsSq, rhsJac, boundaries, borders):
        Result = []
        for i, x in enumerate(np.linspace(boundaries[0][0], boundaries[0][1], self.xGridSize)):
            for j, y in enumerate(np.linspace(boundaries[1][0], boundaries[1][1], self.yGridSize)):
                Result.append(optimize.root(rhs, [x, y], method='broyden1', jac=rhsJac).x)
        allEquilibria = [x for x in Result if abs(rhsSq(x)) < self.eps and inBounds(x, borders)];
        return allEquilibria


class NewtonEqFinderUp:
    def __init__(self, xGridSize, yGridSize, eps):
        self.xGridSize = xGridSize
        self.yGridSize = yGridSize
        self.eps = eps

    def test(self, rhs, x, y, step):
        res = 0
        if rhs((x, y))[0] * rhs((x, y + step))[0] < 0:
            res = 1
        elif rhs((x, y + step))[0] * rhs((x + step, y + step))[0] < 0:
            res = 1
        elif rhs((x + step, y + step))[0] * rhs((x + step, y))[0] < 0:
            res = 1
        elif rhs((x + step, y))[0] * rhs((x, y))[0] < 0:
            res = 1
        if res:
            if rhs((x, y))[1] * rhs((x, y + step))[1] < 0:
                res = 1
            elif rhs((x, y + step))[1] * rhs((x + step, y + step))[1] < 0:
                res = 1
            elif rhs((x + step, y + step))[1] * rhs((x + step, y))[1] < 0:
                res = 1
            elif rhs((x + step, y))[1] * rhs((x, y))[1] < 0:
                res = 1
        return res

    def __call__(self, rhs, rhsSq, rhsJac, boundaries, borders):
        rectangles = np.zeros((self.xGridSize - 1, self.yGridSize - 1))

        x = boundaries[0][0]
        step = (boundaries[0][1] - boundaries[0][0]) / (self.yGridSize - 1)
        for i in range(self.xGridSize - 1):
            y = boundaries[1][0]
            for j in range(self.yGridSize - 1):
                if self.test(rhs, x, y, step):
                    rectangles[self.xGridSize - i - 2][self.yGridSize - j - 2] = 1
                y += step
            x += step

        Result = []
        for i in range(self.xGridSize):
            for j in range(self.yGridSize):
                if rectangles[self.xGridSize - i - 2][self.yGridSize - j - 2]:
                    Result.append(optimize.root(rhs, [boundaries[0][0] + i * step, boundaries[1][0] + j * step],
                                                method='broyden1', jac=rhsJac).x)
        allEquilibria = [x for x in Result if abs(rhsSq(x)) < self.eps and inBounds(x, borders)]
        return allEquilibria


def getEquilibriumInfo(pt, rhsJac):
    eigvals, eigvecs = LA.eig(rhsJac(pt))
    vecs = []
    for i in range(len(eigvals)):
        vecs.append(eigvecs[:, i])
    return Equilibrium(pt, eigvals, vecs)


def createEqList(allEquilibria, rhsJac, ps: PrecisionSettings):
    if len(allEquilibria) > 1:
        allEquilibria = sorted(allEquilibria, key=lambda ar: tuple(ar))
    EqList = []
    for eqCoords in allEquilibria:
        EqList.append(getEquilibriumInfo(eqCoords, rhsJac))
    # подумать, может всё-таки фильтрацию по близости/типу
    # сделать отдельной функцией??
    if len(EqList) > 1:
        trueStr = filterEq(EqList, ps)
        trueEqList = [EqList[i] for i in list(trueStr)]
        return trueEqList

    return EqList


def findEquilibria(rhs, rhsJac, bounds, borders, optMethod, ps: PrecisionSettings):
    def rhsSq(x):
        xArr = np.array(x)
        vec = rhs(xArr)
        return np.dot(vec, vec)

    allEquilibria = optMethod(rhs, rhsSq, rhsJac, bounds, borders)
    return createEqList(allEquilibria, rhsJac, ps)


def inBounds(X, boundaries):
    x, y = X
    Xb, Yb = boundaries
    b1, b2 = Xb
    b3, b4 = Yb
    return ((x > b1) and (x < b2) and (y > b3) and (y < b4))


def filterEq(listEquilibria, ps: PrecisionSettings):
    X = []
    data = []
    for eq in listEquilibria:
        X.append(eq.coordinates)
        data.append(eq.getEqType(ps))
    clustering = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage='single',
                                         distance_threshold=(ps.clustDistThreshold))
    clustering.fit(X)
    return indicesUniqueEq(clustering.labels_, data)


def writeToFileEqList(envParams: EnvironmentParameters, EqList, params, nameOfFile):
    sol = []
    for eq in EqList:
        sol.append(eq.strToFile())
    headerStr = ('gamma = {par[0]}\n' +
                 'd = {par[1]}\n' +
                 'X  Y  nS  nC  nU  isSComplex  isUComplex  Re(eigval1)  Im(eigval1)  Re(eigval2)  Im(eigval2)\n' +
                 '0  1  2   3   4   5           6           7            8            9            10').format(
        par=params)
    fmtList = ['%+18.15f',
               '%+18.15f',
               '%2u',
               '%2u',
               '%2u',
               '%2u',
               '%2u',
               '%+18.15f',
               '%+18.15f',
               '%+18.15f',
               '%+18.15f', ]
    np.savetxt("{env.pathToOutputDirectory}{}.txt".format(nameOfFile, env=envParams), sol, header=headerStr,
               fmt=fmtList)


def createBifurcationDiag(envParams: EnvironmentParameters, numberValuesParam1, numberValuesParam2, arrFirstParam,
                          arrSecondParam):
    N, M = numberValuesParam1, numberValuesParam2
    colorGrid = np.zeros((M, N)) * np.NaN
    diffTypes = {}
    curTypeNumber = 0
    for i in range(M):
        for j in range(N):
            data = np.loadtxt('{}{:0>5}_{:0>5}.txt'.format(envParams.pathToOutputDirectory, i, j), usecols=(2, 3, 4));
            curPhPortrType = describePortrType(data.tolist())
            if curPhPortrType not in diffTypes:
                diffTypes[curPhPortrType] = curTypeNumber
                curTypeNumber += 1.
            colorGrid[j][i] = diffTypes[curPhPortrType]
    plt.pcolormesh(arrFirstParam, arrSecondParam, colorGrid, cmap=plt.cm.get_cmap('RdBu'))
    plt.colorbar()
    plt.savefig('{}{}.pdf'.format(envParams.pathToOutputDirectory, envParams.imageStamp))


def indicesUniqueEq(connectedPoints, nSnCnU):
    arrDiffPoints = {}
    for i in range(len(connectedPoints)):
        pointParams = np.append(nSnCnU[i], connectedPoints[i])
        if tuple(pointParams) not in arrDiffPoints:
            arrDiffPoints[tuple(pointParams)] = i
    return arrDiffPoints.values()


def valP(sdlFocEq, saddlEq, ps: PrecisionSettings):
    sdlLeadingSRe = saddlEq.getLeadSEigRe(ps)
    sdlLeadingURe = saddlEq.getLeadUEigRe(ps)
    sdlFocLeadSRe = sdlFocEq.getLeadSEigRe(ps)
    sdlFocLeadURe = sdlFocEq.getLeadUEigRe(ps)
    p = (-sdlLeadingURe / sdlLeadingSRe) * (-sdlFocLeadURe / sdlFocLeadSRe)
    return p


def embedPointBack(ptOnPlane):
    return [0] + ptOnPlane


def isPtInUpperTriangle(ptOnPlane):
    x, y = ptOnPlane
    return (x <= y)


def isStable2DFocus(eq, ps: PrecisionSettings):
    return eq.getEqType(ps) == [2, 0, 0, 1, 0]


def is3DSaddleFocusWith1dU(eq, ps: PrecisionSettings):
    return eq.getEqType(ps) == [2, 0, 1, 1, 0]


def is2DSaddle(eq, ps: PrecisionSettings):
    return eq.getEqType(ps) == [1, 0, 1, 0, 0]


def is3DSaddleWith1dU(eq, ps: PrecisionSettings):
    return eq.getEqType(ps) == [2, 0, 1, 0, 0]


def has1DUnstable(eq, ps: PrecisionSettings):
    return eq.getEqType(ps)[2] == 1


def goodConfEqList(EqList, rhs, ps: PrecisionSettings):
    sadFocs = []
    saddles = []
    for eq in EqList:
        ptOnInvPlane = eq.coordinates
        ptOnPlaneIn3D = embedPointBack(ptOnInvPlane)
        eqOnPlaneIn3D = getEquilibriumInfo(ptOnPlaneIn3D, rhs.getReducedSystemJac)
        if (isPtInUpperTriangle(ptOnInvPlane)):
            if (isStable2DFocus(eq, ps) and is3DSaddleFocusWith1dU(eqOnPlaneIn3D, ps)):
                sadFocs.append(eqOnPlaneIn3D)
            elif (is2DSaddle(eq, ps) and is3DSaddleWith1dU(eqOnPlaneIn3D, ps)):
                saddles.append(eqOnPlaneIn3D)
    conf = []
    for sf in sadFocs:
        arrSd = []
        for sd in saddles:
            if valP(sf, sd, ps) > 1.:
                arrSd.append(sd)
        if len(arrSd) > 0:
            conf.append([sf, arrSd])
    return conf


def T(X):
    x, y, z = X
    return [y - x, z - x, 2 * np.pi - x]


def generateSymmetricPoints(pt):
    return [pt, T(pt), T(T(pt)), T(T(T(pt)))]


def getInitPointsOnUnstable1DSeparatrix(eq, ps: PrecisionSettings):
    if has1DUnstable(eq, ps):
        unstVector = eq.eigvectors[-1]
        pt1 = eq.coordinates + unstVector * ps.separatrixShift
        pt2 = eq.coordinates - unstVector * ps.separatrixShift
        return [pt1, pt2]
    else:
        raise (ValueError, 'Not a saddle with 1d unstable manifold!')


def isInCIR(pt):
    th2, th3, th4 = pt
    return (0 - 1e-7 <= th2) and (th2 <= th3) and (th3 <= th4) and (th4 <= (2 * np.pi + 1e-7))


def computeSeparatrix(eq: Equilibrium, rhs, ps: PrecisionSettings, maxTime, condition = None, numberSep=None):
    pts = getInitPointsOnUnstable1DSeparatrix(eq, ps)
    if condition:
        for pt in pts:
            if condition(pt):
                startPt = pt.real
    else:
        startPt = pts[numberSep].real

    rhs_vec = lambda t, X: rhs(X) #getReducedSystem(X)
    sol = solve_ivp(rhs_vec, [0, maxTime], startPt, rtol=ps.rTol, atol=ps.aTol, dense_output=True)
    coordPtSeparat = []
    for i in range(len(sol.y[0])):
        coordPtSeparat.append(sol.y[:, i])
    return coordPtSeparat

def createListofAllSymmEq(listOfEq: list[Equilibrium]):
    listOfAllEq = []
    for sd in listOfEq:
        listOfAllEq += generateSymmetricPoints(sd.coordinates)
    return listOfAllEq

def heterCheck(Eq: Equilibrium, listOfEq: list[Equilibrium], rhs, maxTime, ps: PrecisionSettings, condition = None, numberSep = None):

    separatrix = computeSeparatrix(Eq, rhs, ps, maxTime, condition= condition, numberSep= numberSep)
    allDistsWithEq = [np.linalg.norm(np.array(pt) - np.array(coordSd)) for pt in separatrix for coordSd in
                      listOfEq]
    minDistWithEq = min(allDistsWithEq)
    return minDistWithEq
