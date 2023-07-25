import systems_fun as sf
import SystOsscills as a4d
import numpy as np
from scipy.integrate import solve_ivp
from os.path import join

import datetime
import plotly.io as pio
import plotly.graph_objects as go

import os
import sys
import yaml


def getSol(params, startPt,  maxTime, tol = 1e-13):
    rhs=a4d.FourBiharmonicPhaseOscillators(params[0],params[1],params[2],params[3])
    rhs_vec = lambda t, X: rhs.getReducedSystem(X)
    sol = solve_ivp(rhs_vec, [0, maxTime], startPt, rtol = tol, atol= tol, dense_output=True, method='DOP853')
    return sol


def plotAttrs(data_attrs, outName, configDict):
    plotSettings = configDict["plotSettings"]
    symmCopies = []
    if plotSettings["symmCopies"]:
        symmCopies = list(map(int, plotSettings["symmCopies"].split(',')))
    widthLine = plotSettings["widthLine"]
    s = np.array([(0, 0, 0), (2 * np.pi, 2 * np.pi, 2 * np.pi), (0, 0, 2 * np.pi), (0, 2 * np.pi, 2 * np.pi)])
    pts = np.array([s[0], s[1], s[2], s[0], s[3], s[2], s[1], s[3]])
    ptsX, ptsY, ptsZ = pts.T

    xs, ys, zs = data_attrs
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, marker=dict(size=0.001), line=dict(
        color="red",
        width=widthLine)))

    xyzT = [sf.T(coords) for coords in zip(xs, ys, zs)]
    xsT, ysT, zsT = zip(*xyzT)

    xyzT2 = [sf.T(coords) for coords in xyzT]
    xsT2, ysT2, zsT2 = zip(*xyzT2)

    xyzT3 = [sf.T(coords) for coords in xyzT2]
    xsT3, ysT3, zsT3 = zip(*xyzT3)

    if 1 in symmCopies:
        fig.add_trace(go.Scatter3d(x=xsT, y=ysT, z=zsT, marker=dict(size=0.001), line=dict(
            color="blue", width=widthLine)))

    if 2 in symmCopies:
        fig.add_trace(go.Scatter3d(x=xsT2, y=ysT2, z=zsT2, marker=dict(size=0.001), line=dict(
            color="green", width=widthLine)))

    if 3 in symmCopies:
        fig.add_trace(go.Scatter3d(x=xsT3, y=ysT3, z=zsT3, marker=dict(size=0.001), line=dict(
            color="magenta", width=widthLine)))

    fig.add_trace(go.Scatter3d(x=ptsX, y=ptsY, z=ptsZ, marker=dict(size=3), line=dict(
        color="orange")))
    fig.update_layout(showlegend=False,
                      scene=dict(xaxis=dict(backgroundcolor="white"), yaxis=dict(backgroundcolor="white"),
                                 zaxis=dict(backgroundcolor="white")))

    camera = plotSettings["cameraParams"]

    fig.update_layout(autosize=False,
                      width=500,
                      height=500,
                      margin=dict(l=0, r=0, b=0, t=0),
                      scene_camera=camera)
    pio.write_image(fig, outName, format=configDict["output"]["outFileExtension"])

configName = sys.argv[1]
assert os.path.isfile(configName), f"Configuration file {os.path.abspath(configName)} does not exist!"

with open(configName, 'r') as f:
    configDict = yaml.load(f, Loader=yaml.loader.SafeLoader)

assert os.path.isdir(configDict["output"]["directory"]), "Output directory does not exist!"

outDir = configDict["output"]["directory"]
fileName = configDict["output"]["mask"]
if configDict["output"]["useTimeStamp"]:
    startTime = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    fileName = fileName + '_' + startTime
outFileExtension = configDict["output"]["outFileExtension"]

outNameImg = join(outDir, f"{fileName}.{outFileExtension}")

if configDict['pathToData']:
    assert os.path.exists(configDict['pathToData']), f"Data file {configDict['data_file_path']} does not exist!"
    traj = np.loadtxt(configDict['pathToData'], unpack=True)
else:
    a, b, r = configDict['params'].values()
    x, y, z = configDict['startPt'].values()
    params = [0.5, a, b, r]
    startPt = [x, y, z]
    maxTime = configDict['maxTime']
    sol = getSol(params, startPt, maxTime)
    traj = sol.y

    outNameDataFile = join(configDict["output"]["directory"], f"{fileName}.txt")
    headerStr = (
            f'alpha = {params[1]}\n' +
            f'beta = {params[2]}\n' +
            f'r = {params[3]}\n' +
            'x  y  z\n' +
            '0  1  2')
    fmtList = ['%+18.15f',
               '%+18.15f',
               '%+18.15f']

    np.savetxt(outNameDataFile, list(zip(*traj)), header=headerStr,
               fmt=fmtList)

plotAttrs(traj, outNameImg, configDict)