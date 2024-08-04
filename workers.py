# all workers must obey this signature:
# config, timeStamp, gridNode

from os.path import join

import pandas as pd
import numpy as np

import plotFun as pf
import systems_fun as sf
import findTHeteroclinic as fth
import scriptUtils as su
import itertools as itls

from grid import gridNodeToUpdate, gridNodeToIndexes, gridNodeToDict, GridPoint
import SystOsscills as a4d
from workers_utils import prepareOutputDir, register

TARGETHETEROCLINICCOLS = ['distTrajToEq', 'integrationTime',
               'startPtX', 'startPtY', 'startPtZ',
               'sadfocPtX', 'sadfocPtY', 'sadfocPtZ',
               'saddlePtX', 'saddlePtY', 'saddlePtZ']

heteroCols = {'targetHeteroclinic': TARGETHETEROCLINICCOLS}

bounds = [(-0.1, +2 * np.pi + 0.1), (-0.1, +2 * np.pi + 0.1)]
bordersEq = [(-1e-15, +2 * np.pi + 1e-15), (-1e-15, +2 * np.pi + 1e-15)]

registry = {
    "worker": {},
    "init": {},
    "post": {}
}


registry['init']['targetHeteroclinic'] = prepareOutputDir


def makeSystem(gridNode, config):
    # extract default parameters
    defParDict = config['defaultSystem']
    a = defParDict['a']
    b = defParDict['b']
    r = defParDict['r']
    # and make default system
    dsys = a4d.FourBiharmonicPhaseOscillators(0.5, a, b, r)
    # update it with values from gridNode
    upd = gridNodeToUpdate(gridNode)
    dsys.setParams(upd)
    return dsys


def makeFinalOutname(config, initResult, outFileExtension, timeStamp, gridNode=None):
    newDir, newMask = initResult
    if gridNode:
        inds = gridNodeToIndexes(gridNode)
        formatIndStrs = [f"{ind:0>{len(str(config['grid'][pname]['steps']-1))}}"
                         for ind, pname in zip(inds, ['first', 'second'])]
        indStr = '_'+'_'.join(formatIndStrs)
    else:
        indStr = ''
    tStampMode = config['output']['useTimestamp']
    tStampStr = ''
    if tStampMode in ['file', 'both']:
        tStampStr = f"_{timeStamp}"

    outName = join(newDir, f"{newMask}{indStr}{tStampStr}.{outFileExtension}")
    return outName

@register(registry, 'worker', 'targetHeteroclinic')
def workerTargetHeteroclinicMap(gridNode: tuple[GridPoint], config, timeStamp, initResult):
    dsys = makeSystem(gridNode, config)
    # get solver data

    nSamp = config['solver']['nSamp']
    nIters = config['solver']['nIters']
    zeroToCompare = config['solver']['zeroToCompare']
    eqf = sf.ShgoEqFinder(nSamp, nIters, zeroToCompare)
    # get task parameters
    ps = su.getPrecisionSettings(config)
    prox = su.getProximitySettings(config)
    evtFlag = config['Parameters']['useEvents']
    maxTime = config['Parameters']['maxTime']
    # do the task
    res = fth.checkTargetHeteroclinic( dsys, bordersEq, bounds, eqf, ps, prox, maxTime, evtFlag)
    gDict = gridNodeToDict(gridNode)
    heterInfo = heteroCols[config['task']]

    infoDict = {inf : None for inf in heterInfo}
    resDict = []

    if res:
        for info in res:
            tempDict = {}
            infoVals = list(itls.chain.from_iterable([[info['dist'], info['integrationTime']], info['stPt'],
                                                      info['alpha'].coordinates, info['omega'].coordinates]))

            for i, keyVal in enumerate(heterInfo):
                tempDict[keyVal] = infoVals[i]

            resDict.append({**gDict, **tempDict})
    else:
        resDict = [{**gDict, **infoDict}]


    return resDict


@register(registry, 'post', 'targetHeteroclinic')
def postTargetHeteroclinicMap(configDict, initResult, workerResult, grid: list[list[GridPoint]], startTime):
    workerResult = list(itls.chain.from_iterable(workerResult))
    # adjust outName
    outTxtName = makeFinalOutname(configDict, initResult, 'txt', startTime)
    df = pd.DataFrame(workerResult)
    gridParamNames = [g[0].name for g in grid]
    indexColumns = [ind for ind, _ in zip(['i', 'j'], gridParamNames)]
    df.sort_values(by=indexColumns)
    if workerResult:
        infoParams = heteroCols[configDict['task']]
        df[indexColumns + gridParamNames + infoParams].to_csv(outTxtName, sep=' ', index=False)

    ext = configDict['output']['imageExtension']
    outImgName = makeFinalOutname(configDict, initResult, ext, None)
    pltDict = configDict['misc']['plotParams']
    sysParams = list(configDict['defaultSystem'].keys())
    if len(grid) == 1:
        paramNameImg = configDict['grid']['first']['caption']
        paramLims = [configDict['grid']['first']['min'], configDict['grid']['first']['max']]
        sortWorkerResult = sorted(workerResult, key=lambda e: e['i'])
        paramName = configDict['grid']['first']['name']
        sysParams.remove(paramName)
        titleImg = "Heteroclinic graph "
        for par in sysParams:
            titleImg += "{} = {} ".format(par, configDict['defaultSystem'][par])
        hetParamVal = [e[paramName] for e in sortWorkerResult if e['distTrajToEq']]

        if(hetParamVal):
            pf.plotHeteroclinicGraph(paramNameImg, titleImg,  outImgName, paramLims, hetParamVal, pltDict)
    elif len(grid) == 2:
        paramXs = grid[0]
        paramYs = grid[1]
        paramXNameImg = configDict['grid']['first']['caption']
        paramYNameImg = configDict['grid']['second']['caption']
        paramXVals = [gn.val for gn in paramXs]
        paramYVals = [gn.val for gn in paramYs]
        hetParamVals = [(e['i'], e['j']) for e in workerResult if e['distTrajToEq']]

        paramName = configDict['grid']['first']['name']
        sysParams.remove(paramName)
        paramName = configDict['grid']['second']['name']
        sysParams.remove(paramName)

        titleImg = "Heteroclinic map "
        for par in sysParams:
            titleImg += "{} = {} ".format(par, configDict['defaultSystem'][par])

        pf.plotHeteroclinicMap(hetParamVals, paramXVals, paramXNameImg, paramYVals, paramYNameImg, outImgName,
                               titleImg, pltDict)
    else:
        raise Exception("There is no method for such grid!")