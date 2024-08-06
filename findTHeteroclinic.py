import systems_fun as sf
from collections import defaultdict
import itertools as itls
from scipy.spatial import distance



def checkSeparatrixConnection(pairsToCheck, ps: sf.PrecisionSettings, proxs: sf.ProximitySettings, rhs, rhsJac, phSpaceTransformer, sepCondition, eqTransformer, sepNumCondition, sepProximity, maxTime, distFunc, tSkip=0, listEqCoords = None):
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
            events = sf.createListOfEvents(alphaEqTr, fullOmegaEqsTr, listEqCoords, ps, proxs, distFunc)
        separatrices, integrTimes = sf.computeSeparatrices(alphaEqTr, rhs, ps, maxTime, sepCondition, tSkip, events)

        if not sepNumCondition(separatrices):
            raise ValueError('Assumption on the number of separatrices is not satisfied')

        for omegaEqTr in fullOmegaEqsTr:
            for i, separatrix in enumerate(separatrices):
                dist = distance.cdist(separatrix, [omegaEqTr.coordinates], distFunc).min()

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