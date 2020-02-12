import os
import subprocess
from collections import namedtuple
import matplotlib.pyplot as plt
from matplotlib import colors as cm
import numpy as np
import scipy.optimize
from numpy import linalg as LA
MapParameters = namedtuple('MapParameters', ['rhs', 'rhsJac', 'valueFirstParam',
                                             'valueSecondParam', 'constParam', 'bounds','optMethodParams', 'bordersEq'])

class EnvironmentParameters:
    """
    Class that stores information about where
    to get source, where to put compiled version
    and where to save output files
    """

    def __init__(self, pathToOutputDirectory, outputStamp, imageStamp):
        assert (os.path.isdir(pathToOutputDirectory)), 'Output directory does not exist!'
        self.pathToOutputDirectory = os.path.join(os.path.normpath(pathToOutputDirectory), '')
        self.outputStamp = outputStamp
        self.imageStamp = imageStamp

    @property
    def clearAllInOutputDirectory(self):
        return os.path.join(self.pathToOutputDirectory, '*')

    @property
    def fullExecName(self):
        return os.path.join(self.pathToOutputDirectory, self.outputExecutableName)

    # prepare environment function: copy from trc_utils and use https://stackoverflow.com/a/12526809
def prepareEnvironment(envParams):
    """
    Clears output directory and copies executable
    """
    # delete all files in directory
    assert isinstance(envParams, EnvironmentParameters)
    clearOutputCommand = 'rm {env.clearAllInOutputDirectory}'.format(env=envParams)
    subprocess.call(clearOutputCommand, shell=True)
    # compile code
    #compileExecCommand = 'g++ {env.pathToSource} -o {env.fullExecName} --std=c++11'.format(env=envParams)
    #subprocess.call(compileExecCommand, shell=True)


def isComplex(Z):
    return (abs(Z.imag)>1e-14)

def describeEqType(eigvals):
    eigvalsS = eigvals[np.real(eigvals) < -1e-14]
    eigvalsU = eigvals[np.real(eigvals) > +1e-14]
    nS = len(eigvalsS)
    nU = len(eigvalsU)
    nC = len(eigvals) - nS - nU
    issc = 1 if nS > 0 and isComplex(eigvalsS[-1]) else 0
    isuc = 1 if nU > 0 and isComplex(eigvalsU[ 0]) else 0
    return (nS,nC,nU,issc,isuc)

def describePortrType(dataEqSignatures):
    nSaddles  = len([eq for eq in dataEqSignatures if eq==[1, 0, 1]])
    nSources  = len([eq for eq in dataEqSignatures if eq==[0, 0, 2]])
    nSinks    = len([eq for eq in dataEqSignatures if eq==[2, 0, 0]])
    nNonRough = len([eq for eq in dataEqSignatures if eq==[1, 1, 0] or eq==[0, 1, 1] or eq==[0, 2, 0]])
    return(nSaddles,nSources,nSinks,nNonRough)

def findEquilibria(rhs,rhsJac, boundaries, optMethodParams,ud,borders):
    nSamples, nIters = optMethodParams
    def rhsSq(x):
        xArr = np.array(x)
        vec = rhs(xArr)
        return np.dot(vec, vec)
    optResult = scipy.optimize.shgo(rhsSq,boundaries,n= nSamples, iters = nIters, sampling_method='sobol');
    allEquilibria = [x for x, val in zip(optResult.xl, optResult.funl) if abs(val)<1e-15 and inBounds(x,borders)];
    allEquilibria = sorted(allEquilibria, key = lambda ar: tuple(ar))
    result = np.zeros([len(allEquilibria),9])
    for k, eqCoords in enumerate(allEquilibria):
        eqJacMatrix = rhsJac(eqCoords, ud)
        eigvals, _ = LA.eig(eqJacMatrix)
        eqTypeData = describeEqType(eigvals)
        eigvals = sorted(eigvals, key=lambda eigvals: eigvals.real)
        #np.append(result,eqCoords)
        #np.append(result,eqTypeData)
        #np.append(result,eigvals)
        result[k,0] = eqCoords[0]
        result[k,1] = eqCoords[1]
        result[k,2] = eqTypeData[0]
        result[k,3] = eqTypeData[1]
        result[k,4] = eqTypeData[2]
        result[k,5] = eqTypeData[3]
        result[k,6] = eqTypeData[4]
        result[k,7] = eigvals[0]
        result[k,8] = eigvals[1]
    return result
def inBounds(X,boundaries):
    x, y = X
    Xb,Yb = boundaries
    b1,b2=Xb
    b3,b4=Yb
    return ((x > b1) and (x < b2) and (y > b3) and (y < b4))

def createFileTopologStructPhasePort(envParams, mapParams,i,j):
    ud = [mapParams.valueFirstParam,mapParams.valueSecondParam] + mapParams.constParam;
    headerStr =('gamma = {par[0]}\n'+
                'd = {par[1]}\n'+
                'X  Y  nS  nC  nU  isSComplex  isUComplex  eigval1  eigval2\n'+
                '0  1  2   3   4   5           6           7        8').format(par=ud)
    rhsCurrent = lambda X: mapParams.rhs(X, ud)
    sol = findEquilibria (rhsCurrent, mapParams.rhsJac, mapParams.bounds, mapParams.optMethodParams,ud,mapParams.bordersEq)
    np.savetxt('{env.pathToOutputDirectory}{:0>5}_{:0>5}.txt'.format(i,j,env=envParams), sol, header=headerStr)


def createBifurcationDiag(envParams,numberValuesParam1,numberValuesParam2,arrFirstParam,arrSecondParam):
    N,M = numberValuesParam1,numberValuesParam2
    colorGrid = np.zeros((M, N ))*np.NaN
    diffTypes = {}
    curTypeNumber = 0
    for i in range(M):
        for j in range(N):
            data = np.loadtxt('{}{:0>5}_{:0>5}.txt'.format(envParams.pathToOutputDirectory,i, j), usecols = (2, 3, 4));
            curPhPortrType = describePortrType(data.tolist())
            if curPhPortrType not in diffTypes:
                diffTypes[curPhPortrType] = curTypeNumber
                curTypeNumber += 1.
            colorGrid[i][j] = diffTypes[curPhPortrType]
    plt.pcolormesh(arrFirstParam,arrSecondParam,colorGrid,cmap=plt.cm.get_cmap('RdBu'))
    plt.colorbar()
    plt.savefig('{}{}.pdf'.format(envParams.pathToOutputDirectory,envParams.imageStamp))