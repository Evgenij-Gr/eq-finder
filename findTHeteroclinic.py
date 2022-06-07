import systems_fun as sf
from collections import defaultdict
import itertools as itls
import SystOsscills as a4d
from scipy.spatial import distance
import TwoPendulumsSystemFun


def checkSeparatrixConnection(pairsToCheck, ps: sf.PrecisionSettings, proxs: sf.ProximitySettings, rhs, rhsJac, phSpaceTransformer, sepCondition, eqTransformer, sepNumCondition, sepProximity, maxTime, periodDistance, listEqCoords = None):
    """
    Accepts pairsToCheck — a list of pairs of Equilibria — and checks if there is
    an approximate connection between them. First equilibrium of pair
    must be a saddle with one-dimensional unstable manifold. The precision of
    connection is given by :param sepProximity.
    """
    grpByAlphaEq = defaultdict(list)
    for alphaEq, omegaEq in pairsToCheck:
        grpByAlphaEq[alphaEq].append(omegaEq)

    outputInfo = []

    events = None

    for alphaEq, omegaEqs in grpByAlphaEq.items():
        alphaEqTr = phSpaceTransformer(alphaEq, rhsJac)
        omegaEqsTr = [phSpaceTransformer(oEq, rhsJac) for oEq in omegaEqs]
        fullOmegaEqsTr = list(itls.chain.from_iterable([eqTransformer(oEq, rhsJac) for oEq in omegaEqsTr]))
        if listEqCoords:
            events = sf.createListOfEvents(alphaEqTr, fullOmegaEqsTr, listEqCoords, ps, proxs)
        separatrices, integrTimes = sf.computeSeparatrices(alphaEqTr, rhs, ps, maxTime, sepCondition, events)

        if not sepNumCondition(separatrices):
            raise ValueError('Assumption on the number of separatrices is not satisfied')

        for omegaEqTr in fullOmegaEqsTr:
            for i, separatrix in enumerate(separatrices):
                if periodDistance == 1:
                    dist = TwoPendulumsSystemFun.Distance4D(separatrix, omegaEqTr.coordinates)
                else:
                    dist = distance.cdist(separatrix, [omegaEqTr.coordinates]).min()

                if dist < sepProximity:
                    info = {}
                    # TODO: what exactly to output
                    info['alpha'] = alphaEqTr
                    info['omega'] = omegaEqTr
                    info['stPt'] = separatrix[0]
                    info['dist'] = dist
                    info['integrationTime'] = integrTimes[i]
                    outputInfo.append(info)

    return outputInfo

def checkTargetHeteroclinic(osc: a4d.FourBiharmonicPhaseOscillators, borders, bounds, eqFinder, ps: sf.PrecisionSettings, proxs: sf.ProximitySettings, maxTime, withEvents = False):
    rhsInvPlane = osc.getRestriction
    jacInvPlane = osc.getRestrictionJac
    rhsReduced = osc.getReducedSystem
    jacReduced = osc.getReducedSystemJac


    planeEqCoords = sf.findEquilibria(rhsInvPlane, jacInvPlane, bounds, borders, eqFinder, ps)

    if withEvents:
        eqCoords3D = sf.listEqOnInvPlaneTo3D(planeEqCoords, osc)
        allSymmEqs = itls.chain.from_iterable([sf.cirTransform(eq, jacReduced) for eq in eqCoords3D])
    else:
        allSymmEqs = None
    tresserPairs = sf.getSaddleSadfocPairs(planeEqCoords, osc, ps, needTresserPairs=True)

    cnctInfo = checkSeparatrixConnection(tresserPairs, ps, proxs, rhsInvPlane, jacInvPlane, sf.idTransform, sf.pickBothSeparatrices, sf.idListTransform, sf.anyNumber, proxs.toSinkPrxty, maxTime, listEqCoords = planeEqCoords)
    newPairs = {(it['omega'], it['alpha']) for it in cnctInfo}
    finalInfo = checkSeparatrixConnection(newPairs, ps, proxs, rhsReduced, jacReduced, sf.embedBackTransform, sf.pickCirSeparatrix, sf.cirTransform, sf.hasExactly(1), proxs.toSddlPrxty, maxTime, listEqCoords = allSymmEqs)
    return finalInfo

def checkSadfoc_SaddleHeteroclinic(osc: a4d.FourBiharmonicPhaseOscillators, borders, bounds, eqFinder, ps: sf.PrecisionSettings, proxs: sf.ProximitySettings, maxTime, withEvents = False):
    rhsInvPlane = osc.getRestriction
    jacInvPlane = osc.getRestrictionJac
    rhsReduced = osc.getReducedSystem
    jacReduced = osc.getReducedSystemJac


    planeEqCoords = sf.findEquilibria(rhsInvPlane, jacInvPlane, bounds, borders, eqFinder, ps)

    if withEvents:
        eqCoords3D = sf.listEqOnInvPlaneTo3D(planeEqCoords, osc)
        allSymmEqs = itls.chain.from_iterable([sf.cirTransform(eq, jacReduced) for eq in eqCoords3D])
    else:
        allSymmEqs = None

    saddleSadfocPairs = sf.getSaddleSadfocPairs(planeEqCoords, osc, ps)
    cnctInfo = checkSeparatrixConnection(saddleSadfocPairs, ps, proxs, rhsInvPlane, jacInvPlane, sf.idTransform, sf.pickBothSeparatrices, sf.idListTransform, sf.anyNumber, proxs.toSinkPrxty, maxTime, listEqCoords = planeEqCoords)
    newPairs = {(it['omega'], it['alpha']) for it in cnctInfo}
    finalInfo = checkSeparatrixConnection(newPairs, ps, proxs, rhsReduced, jacReduced, sf.embedBackTransform, sf.pickCirSeparatrix, sf.cirTransform, sf.hasExactly(1), proxs.toSddlPrxty, maxTime, listEqCoords = allSymmEqs)

    return finalInfo

def checkTargetHeteroclinicInInterval(osc: a4d.FourBiharmonicPhaseOscillators, borders, bounds, eqFinder, ps: sf.PrecisionSettings, proxs: sf.ProximitySettings, maxTime, lowerLimit):
    info = checkTargetHeteroclinic(osc, borders, bounds, eqFinder, ps, proxs, maxTime)
    finalInfo = []
    for dic in info:
        if dic['dist'] > lowerLimit:
            finalInfo.append(dic)
    return finalInfo

def getStartPtsForLyapVals(osc: a4d.FourBiharmonicPhaseOscillators, borders, bounds, eqFinder, ps: sf.PrecisionSettings, OnlySadFoci):
    rhsInvPlane = osc.getRestriction
    jacInvPlane = osc.getRestrictionJac
    planeEqCoords = sf.findEquilibria(rhsInvPlane, jacInvPlane, bounds, borders, eqFinder, ps)
    ListEqs1dU = sf.get1dUnstEqs(planeEqCoords, osc, ps, OnlySadFoci)
    outputInfo = []

    for eq in ListEqs1dU:
        ptOnInvPlane = eq.coordinates
        ptOnPlaneIn3D = sf.embedPointBack(ptOnInvPlane)
        eqOnPlaneIn3D = sf.getEquilibriumInfo(ptOnPlaneIn3D, osc.getReducedSystemJac)
        startPts = sf.getInitPointsOnUnstable1DSeparatrix(eqOnPlaneIn3D,sf.pickCirSeparatrix, ps)[0]

        outputInfo.append(startPts)
    return outputInfo

def getTresserPairs(osc: a4d.FourBiharmonicPhaseOscillators, borders, bounds, eqFinder, ps: sf.PrecisionSettings):
    rhsInvPlane = osc.getRestriction
    jacInvPlane = osc.getRestrictionJac

    planeEqCoords = sf.findEquilibria(rhsInvPlane, jacInvPlane, bounds, borders, eqFinder, ps)

    tresserPairs = sf.getSaddleSadfocPairs(planeEqCoords, osc, ps, needTresserPairs=True)

    return tresserPairs