import numpy as np


class GridPoint:
    def __init__(self, i, name, val):
        self.idx = i
        self.name = name
        self.val = val


def getGrid(configDict):
    paramOrderKeys = ['first', 'second']
    gridKeys = configDict['grid'].keys()
    if set(paramOrderKeys) == set(gridKeys) or {'first'} == set(gridKeys):
        params = []
        for key in paramOrderKeys:
            if key in gridKeys:
                data = configDict['grid'][key]
                vals = np.linspace(data['min'], data['max'], data['steps'])
                curGrid = [GridPoint(i, data['name'], v) for i, v in enumerate(vals)]
                params.append(curGrid)
        return params
    else:
        raise KeyError(f"Wrong configuration: must have either both 'first' and 'second', or just 'first' as keys")


def gridNodeToUpdate(gridNode: tuple[GridPoint]):
    """ Expects a tuple or two of GridPoint type
    """
    updateDict = {}
    for gn in gridNode:
        updateDict[gn.name] = gn.val
    return updateDict


def gridNodeToIndexes(gridNode: tuple[GridPoint]):
    """ Expects a single or two of tuples of format
        (index, varName, varValue)
    """
    indList = []
    for gp in gridNode:
        indList.append(gp.idx)
    return tuple(indList)


def gridNodeToDict(gridNode: tuple[GridPoint]):
    upd = gridNodeToUpdate(gridNode)
    inds = gridNodeToIndexes(gridNode)
    # there is a trick here: even if inds
    # has only length 1, the zip is correct
    # and takes only first element
    indDict = dict(zip(['i', 'j'], inds))
    return {**upd, **indDict}
