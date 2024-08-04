from os import mkdir
from os.path import join, isdir

def register(registry, jobType, *names):
    def decoratorRegisterJob(func):
        if jobType in ['worker', 'init', 'post']:
            for name in names:
                registry[jobType][name] = func
        else:
            raise KeyError("type must be either 'worker', 'init' or 'post'")
        return func

    return decoratorRegisterJob

# worker always generates data or does some stuff
# mostly we generate data and gather it in the end
# postprocessing manages that data and does final step
def workerIdle(gridNode, config, timeStamp, initResult):
    return None


def workerWatch(gridNode, config, timeStamp, initResult):
    return gridNode, timeStamp, config['task']


# init functions are used to load some stuff
# that would be used by all workers: for example,
# a file with starting points should be loaded only once
# or creates a directory that every worker would use
def initIdle(config, timeStamp):
    return None


# usually it saves generated data and plots something
# basing on it
def postIdle(config, initResult, workerResult, grid, timeStamp):
    return None


def prepareOutputDir(config, timeStamp):
    outputDict = config['output']
    targetDir = outputDict['directory']
    mask = outputDict['mask']

    # is it really neeeded? can we mkdir without this existing?
    assert isdir(targetDir), f"Directory {targetDir} does not exist!"

    stampMode = outputDict['useTimestamp']
    if stampMode in ['folder', 'both']:
        newTargetDir = join(targetDir, timeStamp)
        mkdir(newTargetDir)
        return newTargetDir, mask
    elif stampMode in ['file', 'ignore']:
        # we'll add mask later
        return targetDir, mask
    else:
        raise KeyError(f"Unknown 'useTimeStamp' key ({stampMode})")


def getTaskParams(config):
    """
        Better make it a separate function
        to never handle that manually
    """
    return config[config['task']]
