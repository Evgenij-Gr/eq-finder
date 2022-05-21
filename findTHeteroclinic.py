import systems_fun as sf
from collections import defaultdict
import numpy as np
import itertools
import itertools as itls
import SystOsscills as a4d
from scipy.spatial import distance


def checkSeparatrixConnection(pairsToCheck, ps: sf.PrecisionSettings, proxs: sf.ProximitySettings, rhs, rhsJac, phSpaceTransformer, sepCondition, eqTransformer, sepNumCondition, sepProximity, maxTime, listEqCoords = None):
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
                dist = distance.cdist(separatrix, [omegaEqTr.coordinates]).min()
                if dist < sepProximity:
                    info = {}
                    # TODO: what exactly to output
                    info['alpha'] = alphaEqTr
                    info['omega'] = omegaEqTr
                    info['stPt']  = separatrix[0]
                    info['dist'] = dist
                    info['integrationTime'] = integrTimes[i]
                    outputInfo.append(info)

    return outputInfo

def checkSourceSinkConnectionOnPlane(pairsToCheck, ps: sf.PrecisionSettings, proxs: sf.ProximitySettings, rhs, rhsJac, phSpaceTransformer, eqTransformer, sepProximity, maxTime, listEqCoords = None):
    """
    """
    grpByAlphaEq = defaultdict(list)
    for alphaEq, omegaEq in pairsToCheck:
        grpByAlphaEq[alphaEq].append(omegaEq)

    outputInfo = []

    events = None
    fis = np.linspace(0, 2 * np.pi, 36)

    for alphaEq, omegaEqs in grpByAlphaEq.items():
        alphaEqTr = phSpaceTransformer(alphaEq, rhsJac)
        omegaEqsTr = [phSpaceTransformer(oEq, rhsJac) for oEq in omegaEqs]
        fullOmegaEqsTr = list(itls.chain.from_iterable([eqTransformer(oEq, rhsJac) for oEq in omegaEqsTr]))
        if listEqCoords:
            events = sf.createListOfEvents(alphaEqTr, fullOmegaEqsTr, listEqCoords, ps, proxs)
        x, y = alphaEqTr.coordinates
        flag = False
        for fi in fis:
            if flag:
                break
            startPt = [x + 0.01*np.cos(fi), y + 0.01*np.sin(fi)]
            traj, integrTime = sf.computeTraj(startPt, rhs, ps, maxTime, events)
            for omegaEqTr in fullOmegaEqsTr:
                dist = distance.cdist(traj, [omegaEqTr.coordinates]).min()
                if dist < sepProximity:
                    flag = True
                    info = {}
                    info['alpha'] = alphaEqTr
                    info['omega'] = omegaEqTr
                    info['stPt'] = startPt
                    info['dist'] = dist
                    info['integrationTime'] = integrTime
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

def checkHeterocninicSf1Sf2SaddleLig(osc: a4d.FourBiharmonicPhaseOscillators, borders, bounds, eqFinder, ps: sf.PrecisionSettings, proxs: sf.ProximitySettings, maxTime, withEvents = False):
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
    saddles, sadFocsWith1dU, sadFocsWith1dS = sf.getSf1Sf2Sad(planeEqCoords, osc, ps)
    finalInfo = []
    cnctInfo = []
    if (saddles and sadFocsWith1dU and sadFocsWith1dS):
        """Check separatrix connection between saddle-focus with 1dU and saddle on edge"""
        pairEqToCnct = itertools.product(saddles, sadFocsWith1dU)
        cnctInfo = checkSeparatrixConnection(pairEqToCnct, ps, proxs, rhsInvPlane, jacInvPlane, sf.idTransform,
                                             sf.pickBothSeparatrices, sf.idListTransform, sf.anyNumber,
                                             proxs.toSinkPrxty, maxTime, listEqCoords=planeEqCoords)

    """Check separatrix connection between saddle-focus With 1dU and saddle-focus With 1dS"""
    if cnctInfo:
        sadSf1dUList = [(it['alpha'], it['omega']) for it in cnctInfo]
        targSfWith1dU = np.array(sadSf1dUList)[:, 1]
        pairEqToCnct = itertools.product(targSfWith1dU, sadFocsWith1dS)
        cnctInfo = checkSeparatrixConnection(pairEqToCnct, ps, proxs, rhsReduced, jacReduced, sf.embedBackTransform,
                                             sf.pickCirSeparatrix, sf.cirTransform, sf.hasExactly(1), proxs.toSddlPrxty,
                                             maxTime, listEqCoords=allSymmEqs)
        sf1dU_sf1dS_List = [(it['alpha'], it['omega'], it['stPt'], it['dist'], it['integrationTime']) for it in cnctInfo]
        sadSf1dUSf1dSList = []
        for sf1dU, sf1dS, stPt, dist, intTime in sf1dU_sf1dS_List:
            for sad, foc in sadSf1dUList:
                if (sf.eqOfEqs(sf1dU, sf.embedBackTransform(foc, jacReduced))):
                    sadSf1dUSf1dSList.append([sf.embedBackTransform(sad, jacReduced), sf1dU, sf1dS, stPt, dist, intTime])

    """Check separatrix connection between saddle-focus With 1dS and saddle on edge"""
    if cnctInfo:
        targSfWith1dS = np.array(sadSf1dUSf1dSList)[:, 2]
        targSfWith1dSOnPlane = [sf.eqCopyOnPlane(eq, jacReduced, ps) for eq in targSfWith1dS]
        targSads = np.array(sadSf1dUSf1dSList)[:, 0]
        targNodes = [sf.getEquilibriumInfo(sf.T(sf.T(eq.coordinates)), jacReduced) for eq in targSads]
        pairEqToCnct = list(zip(targSfWith1dSOnPlane, targNodes))
        cnctInfo = checkSourceSinkConnectionOnPlane(pairEqToCnct, ps, proxs, rhsInvPlane, jacInvPlane,
                                                    sf.planeTransform,
                                                    sf.idListTransform, proxs.toSinkPrxty,
                                                    maxTime, listEqCoords=planeEqCoords)
        sf1dS_saddle_List = {(it['alpha'], it['omega']) for it in cnctInfo}
        for sf1dSPlane, sadPlane in sf1dS_saddle_List:
            for sad, sf1dU, sf1dS, stPt, dist, intTime in sadSf1dUSf1dSList:
                if (sf.eqOfEqs(sf.planeTransform(sf.eqCopyOnPlane(sf1dS, jacReduced, ps), jacInvPlane), sf1dSPlane) and
                        sf.eqOfEqs(sf.planeTransform(sf.getEquilibriumInfo(sf.T(sf.T(sad.coordinates)), jacReduced),
                                                     jacInvPlane), sadPlane)):
                    finalInfo.append([sad, sf1dU, sf1dS, stPt, dist, intTime])

    return finalInfo

def checkHeterocninicNet(osc: a4d.FourBiharmonicPhaseOscillators, borders, bounds, eqFinder, ps: sf.PrecisionSettings, proxs: sf.ProximitySettings, maxTime, withEvents = False):
    rhsInvPlane = osc.getRestriction
    jacInvPlane = osc.getRestrictionJac
    rhsReduced = osc.getReducedSystem
    jacReduced = osc.getReducedSystemJac
    rhsInvPlaneRev = osc.getRestrictionRev
    jacInvPlaneRev = osc.getRestrictionJacRev

    planeEqCoords = sf.findEquilibria(rhsInvPlane, jacInvPlane, bounds, borders, eqFinder, ps)

    if withEvents:
        eqCoords3D = sf.listEqOnInvPlaneTo3D(planeEqCoords, osc)
        allSymmEqs = itls.chain.from_iterable([sf.cirTransform(eq, jacReduced) for eq in eqCoords3D])
    else:
        allSymmEqs = None

    saddles, sadFocsWith1dS = sf.getSadSf(planeEqCoords, osc, ps)
    sadRev = [sf.getEquilibriumInfo(eq.coordinates, jacInvPlaneRev) for eq in saddles]
    sadfocRev = [sf.getEquilibriumInfo(eq.coordinates, jacInvPlaneRev) for eq in sadFocsWith1dS]
    cnctInfo = []

    if len(sadRev) > 1 and sadfocRev:
        #print(f"Число комбинаций {len(sadRev)*len(sadfocRev)}\n Число седел {len(sadRev)} \n Число фокусов {len(sadfocRev)}\n Параметры {osc.getParams()}\n")
        if(len(sadRev)*len(sadfocRev) < 1000):
            pairEqToCnct = itertools.product(sadRev, sadfocRev)
            cnctInfo = checkSeparatrixConnection(pairEqToCnct, ps, proxs, rhsInvPlaneRev, jacInvPlaneRev,
                                                 sf.idTransform, sf.pickBothSeparatrices, sf.idListTransform,
                                                 sf.anyNumber, proxs.toSinkPrxty, maxTime, listEqCoords=planeEqCoords)


    # newPairs = {(sf.getEquilibriumInfo(it['alpha'].coordinates, jacInvPlane),
    #              sf.getEquilibriumInfo(it['omega'].coordinates, jacInvPlane)) for it in cnctInfo}
    #
    # finalInfo = checkSeparatrixConnection(newPairs, ps, proxs, rhsReduced, jacReduced, sf.embedBackTransform,
    #                                       sf.pickCirSeparatrix, sf.cirTransform, sf.hasExactly(1), proxs.toSddlPrxty,
    #                                       maxTime, listEqCoords=allSymmEqs)
    #
    # if(len(finalInfo) < 2):
    #     finalInfo = []

    return cnctInfo