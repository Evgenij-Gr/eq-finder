import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

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