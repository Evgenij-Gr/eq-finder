using Distributed
# Usage: julia classificationSymmetry.jl <pathToDataFile> <outputMask> <outputDir>
addprocs(8)

@everywhere using PyCall
@everywhere np =  pyimport("numpy")
@everywhere include("funForJuliaScripts.jl")
@everywhere using Main.FunForJulia
@everywhere using YAML
@everywhere configDict = YAML.load_file($(ARGS[1]))


@everywhere headerStr_lyapVals = ("i  j  alpha  beta  r  LyapunovVal_1  LyapunovVal_2  LyapunovVal_3\n0  1  2      3     4  5              6              7")
@everywhere fmtList_lyapVals = ["%2u", "%2u", "%+18.15f", "%+18.15f", "%+18.15f", "%+18.15f", "%+18.15f", "%+18.15f",]
@everywhere headerStr_classSym = ("i  j  alpha  beta  r  ClassOfSymmetry\n0  1  2      3     4  5")
@everywhere fmtList_classSym = ["%s", "%s", "%18s", "%18s", "%s", "%s",]
@everywhere headerStr_calcStartPts = ("i  j  alpha  beta  r  ptX ptY ptZ infNormRHS minDistToEdge minDistToSpSt minDistToPlank\n0  1  2      3     4  5   6   7   8       9       10         11")
@everywhere fmtList_calcStartPts = ["%2u", "%2u", "%+18.15f", "%+18.15f", "%+18.15f", "%+18.15f", "%+18.15f", "%+18.15f", "%+18.15f", "%+18.15f", "%+18.15f", "%+18.15f"]

@everywhere utils_for_tasks = Dict([
("lyapVals", Dict([("taskFun", getLyapunovData),
 ("headerStr", headerStr_lyapVals), ("ftmList", fmtList_lyapVals)])),
 ("classSymm", Dict([("taskFun", getClassOfSymm),
 ("headerStr", headerStr_classSym), ("ftmList", fmtList_classSym)])),
  ("calcStartPts", Dict([("taskFun", getPtOnAttr),
 ("headerStr", headerStr_calcStartPts), ("ftmList", fmtList_calcStartPts)])),
 ("pullingAttr", Dict([("taskFun", getPtOnAttr),
 ("headerStr", headerStr_calcStartPts), ("ftmList", fmtList_calcStartPts)]))])

headerStr = utils_for_tasks[configDict["task"]]["headerStr"]
fmtList = utils_for_tasks[configDict["task"]]["ftmList"]

using Dates
timeOfRun = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
outputFileMask = configDict["output"]["mask"]
pathToOutputDir = configDict["output"]["directory"]

if configDict["output"]["useTimestamp"]
    outputFileMask = string(outputFileMask, "_", timeOfRun)
end

outputFileMask = string(outputFileMask, ".txt")
outputFile = string(pathToOutputDir, outputFileMask)
if configDict["task"] in ["lyapVals", "classSymm", "calcStartPts"]
    @everywhere data = np.loadtxt(configDict["pathToData"])
    @everywhere data = prepareData(data)

    @time begin
        result = pmap(utils_for_tasks[configDict["task"]]["taskFun"], data)
        end
    if !isempty(result)
                np.savetxt(outputFile, result[1,:], header=headerStr, fmt=fmtList)
    end
end

if configDict["task"] in ["pullingAttr",]
    maxTime = configDict["taskParams"]["maxTime"]
    evalTs = configDict["taskParams"]["evalTs"]
    sData = initGrid(configDict["taskParams"])
    @time parallelPullingAttractors(sData, configDict["taskParams"]["a_part"], configDict["taskParams"]["b_part"], maxTime, evalTs)
    resData = []
    ds = 9 # number of columns
    for i in 1:(configDict["taskParams"]["a_part"]*configDict["taskParams"]["b_part"])
        append!(resData, [sData[(i-1)*ds+1:i*ds]])
    end
    np.savetxt(outputFile, resData, header=headerStr, fmt=fmtList)
end