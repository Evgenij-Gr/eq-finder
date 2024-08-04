import datetime
import multiprocessing as mp
import sys
import os.path
import time
from itertools import product
import os

import yaml

import workers as wrk
import workers_utils as wrkut

import grid
from functools import partial

# wrk.register['init']['watch'] = wrkut.initIdle
# wrk.registryWorkers['watch'] = wrkut.workerIdle
# wrk.registryPost['watch'] = wrkut.postIdle

if __name__ == "__main__":
    # print(os.getcwd())
    if '-h' in sys.argv or '--help' in sys.argv:
        print("Usage: python compute.py <pathToConfig>"
              "\n    pathToConfig: full path to configuration file (e.g., \"./cfg.yaml\")")
        sys.exit()

    configName = sys.argv[1]
    assert os.path.isfile(configName), f"Configuration file {os.path.abspath(configName)} does not exist!"

    with open(configName, 'r') as f:
        configDict = yaml.load(f, Loader=yaml.loader.SafeLoader)

    startTime = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    taskName = configDict['task']
    initFunc = wrk.registry['init'].get(taskName, wrkut.initIdle)
    initResult = initFunc(configDict, startTime)

    # define the grid
    grid = grid.getGrid(configDict)
    # define the worker
    # worker must be present
    worker = wrk.registry['worker'][taskName]

    pool = mp.Pool(mp.cpu_count())
    start = time.time()
    workerResult = pool.map(partial(worker, config=configDict, timeStamp=startTime, initResult=initResult), product(*grid))
    end = time.time()
    pool.close()

    # for e in workerResult:
    #     print(e)

    # define the post processing
    postProcess = wrk.registry['post'].get(taskName, wrkut.postIdle)
    postProcess(configDict, initResult, workerResult, grid, startTime)

