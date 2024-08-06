import scipy
import numpy as np

from scipy import optimize
from numpy import linalg as LA
from sklearn.cluster import AgglomerativeClustering
from scipy.integrate import solve_ivp


class PrecisionSettings:
    def __init__(self, zeroImagPartEps, zeroRealPartEps, clustDistThreshold, separatrixShift, separatrix_rTol,
                 separatrix_aTol, marginBorder):
        assert zeroImagPartEps > 0, "Precision must be greater than zero!"
        assert zeroRealPartEps > 0, "Precision must be greater than zero!"
        assert clustDistThreshold > 0, "Precision must be greater than zero!"
        assert separatrixShift > 0, "Precision must be greater than zero!"
        assert separatrix_rTol > 0, "Precision must be greater than zero!"
        assert separatrix_aTol > 0, "Precision must be greater than zero!"

        self.zeroImagPartEps = zeroImagPartEps
        self.zeroRealPartEps = zeroRealPartEps
        self.clustDistThreshold = clustDistThreshold
        self.separatrixShift = separatrixShift
        self.rTol = separatrix_rTol
        self.aTol = separatrix_aTol
        self.marginBorder = marginBorder

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
                                  separatrix_aTol=1e-11,
                                  marginBorder=0
                                  )

class ProximitySettings:
    def __init__(self, toSinkPrxtyEv, toSddlPrxtyEv, toTargetSinkPrxtyEv, toTargetSddlPrxtyEv, toSinkPrxty, toSddlPrxty):
        assert toSinkPrxtyEv > 0, "Precision must be greater than zero!"
        assert toSddlPrxtyEv > 0, "Precision must be greater than zero!"
        assert toTargetSinkPrxtyEv > 0, "Precision must be greater than zero!"
        assert toTargetSddlPrxtyEv > 0, "Precision must be greater than zero!"
        assert toSinkPrxty > 0, "Precision must be greater than zero!"
        assert toSddlPrxty > 0, "Precision must be greater than zero!"

        self.toSinkPrxtyEv = toSinkPrxtyEv
        self.toSddlPrxtyEv = toSddlPrxtyEv
        self.toTargetSinkPrxtyEv = toTargetSinkPrxtyEv
        self.toTargetSddlPrxtyEv = toTargetSddlPrxtyEv
        self.toSinkPrxty = toSinkPrxty
        self.toSddlPrxty = toSddlPrxty

STD_PROXIMITY = ProximitySettings(toSinkPrxtyEv=1e-6,
                                  toSddlPrxtyEv=1e-3,
                                  toTargetSinkPrxtyEv=9*1e-6,
                                  toTargetSddlPrxtyEv=2*1e-3, #тут было 9e-3
                                  toSinkPrxty=1e-5,
                                  toSddlPrxty=1e-2
                                  )

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


class ShgoEqFinder:
    def __init__(self, nSamples, nIters, eps):
        self.nSamples = nSamples
        self.nIters = nIters
        self.eps = eps

    def __call__(self, rhs, rhsJac, eqRhs, eqJac, boundaries, borders):
        def eqRhsSquared(x):
            xArr = np.array(x)
            vec = eqRhs(xArr)
            return np.dot(vec, vec)

        optResult = scipy.optimize.shgo(eqRhsSquared, boundaries, n=self.nSamples, iters=self.nIters, sampling_method='sobol');
        allEquilibria = [x for x, val in zip(optResult.xl, optResult.funl) if
                         abs(val) < self.eps and inBounds(x, borders)]
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


def findEquilibria(rhs, rhsJac, eqRhs, eqJac, embedInPhaseSpace, bounds, borders, optMethod, ps: PrecisionSettings):
    allEqCoords = optMethod(rhs, rhsJac, eqRhs, eqJac, bounds, borders)
    allEquilibria = list(map(embedInPhaseSpace, allEqCoords))
    return createEqList(allEquilibria, rhsJac, ps)


def inBounds(X, boundaries):
    flag = True
    for i, borders in enumerate(boundaries):
        if (X[i] <= borders[0]) or (X[i] >= borders[1]):
            flag = False
    return flag


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


def indicesUniqueEq(connectedPoints, nSnCnU):
    arrDiffPoints = {}
    for i in range(len(connectedPoints)):
        pointParams = np.append(nSnCnU[i], connectedPoints[i])
        if tuple(pointParams) not in arrDiffPoints:
            arrDiffPoints[tuple(pointParams)] = i
    return arrDiffPoints.values()


def embedPointBack(ptOnPlane):
    return [0] + ptOnPlane


def is4DSaddleFocusWith1dU(eq, ps: PrecisionSettings):
    return eq.getEqType(ps) == [3, 0, 1, 1, 0]


def has1DUnstable(eq, ps: PrecisionSettings):
    return eq.getEqType(ps)[2] == 1


def getInitPointsOnUnstable1DSeparatrix(eq, condition, ps: PrecisionSettings):
    if has1DUnstable(eq, ps):
        unstVector = eq.eigvectors[-1]
        pt1 = (eq.coordinates + unstVector * ps.separatrixShift).real
        pt2 = (eq.coordinates - unstVector * ps.separatrixShift).real
        allStartPts = [pt1, pt2]
        return [pt for pt in allStartPts if condition(pt, eq.coordinates)]
    else:
        raise ValueError('Not a saddle with 1d unstable manifold!')


def pickBothSeparatrices(ptCoord, eqCoord):
    return True


def computeSeparatrices(eq: Equilibrium, rhs, ps: PrecisionSettings, maxTime, condition, tSkip, listEvents = None):
    startPts = getInitPointsOnUnstable1DSeparatrix(eq, condition, ps)
    rhs_vec = lambda t, X: rhs(X)
    separatrices = []
    integrationTime = []
    for startPt in startPts:
        sol = solve_ivp(rhs_vec, [0, maxTime], startPt, events=listEvents, rtol=ps.rTol, atol=ps.aTol,
                        dense_output=True)
        # print(max(sol.t))
        # print(tSkip)
        coordsList = list(zip(*sol.y))
        sepPart = [y for y, t in list(zip(coordsList, sol.t)) if t>tSkip]
        # print(sepPart)
        separatrices.append(sepPart)
        # separatrices.append(np.transpose(sol.y))
        integrationTime.append(sol.t[-1])
    return [separatrices, integrationTime]

def constructDistEvent(x0, eps, distFunc):
    evt = lambda t, X: distFunc(x0, X) - eps
    return evt

def isSaddle(eq, ps: PrecisionSettings):
    eqType = eq.getEqType(ps)
    return eqType[0] > 0 and eqType[1]==0 and eqType[2] > 0

def isSink(eq, ps: PrecisionSettings):
    eqType = eq.getEqType(ps)
    return eqType[1] == 0 and eqType[2] == 0


def createListOfEvents(startEq, targetEqs, eqList, ps: PrecisionSettings, proxs: ProximitySettings, distFunc):
    listEvents = []

    for eq in eqList:
        if eq.coordinates != startEq.coordinates:
            isTargetEq = False

            coords = eq.coordinates

            for targetEq in targetEqs:
                if targetEq.coordinates == eq.coordinates:
                    isTargetEq = True

            if isSaddle(eq, ps):
                if isTargetEq:
                    event = constructDistEvent(coords, proxs.toTargetSddlPrxtyEv, distFunc)
                else:
                    event = constructDistEvent(coords, proxs.toSddlPrxtyEv, distFunc)
                event.terminal = True
                event.direction = -1
                listEvents.append(event)
            elif isSink(eq, ps):
                if isTargetEq:
                    event = constructDistEvent(coords, proxs.toTargetSinkPrxtyEv, distFunc)
                else:
                    event = constructDistEvent(coords, proxs.toSinkPrxtyEv, distFunc)
                event.terminal = True
                event.direction = -1
                listEvents.append(event)

    return listEvents


def idTransform(X, rhsJac):
    """
    Accepts an Equilibrium and returns it as is
    """
    return X

def idListTransform(X, rhsJac):
    """
    Accepts an Equilibrium and returns it wrapped in list
    """
    return [X]


def anyNumber(seps):
    return True

