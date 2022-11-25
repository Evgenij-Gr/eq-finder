using Distributed
# Usage: julia LyapVals.jl <pathToDataFile> <outputMask> <outputDir>
addprocs(8)
@everywhere using PyCall
@everywhere np =  pyimport("numpy")
@everywhere attrData = np.loadtxt($(ARGS[1]))
@everywhere include("funForJuliaScripts.jl")
@everywhere using Main.FunForJulia

@everywhere data = prepareData(attrData)

using Dates
timeOfRun = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
nameOutputFile = ARGS[2]
pathToOutputDir = ARGS[3]
outputFileMask = string(nameOutputFile, timeOfRun)
outputFileMask = string(outputFileMask, ".txt")
OutputFile = string(pathToOutputDir, outputFileMask )
@time begin
    result = pmap(getLyapunovData,  data)
end

if !isempty(result)
        headerStr = (
                "i  j  alpha  beta  r  LyapunovVal_1  LyapunovVal_2  LyapunovVal_3\n0  1  2      3     4  5              6              7")
        fmtList = ["%2u",
                   "%2u",
                   "%+18.15f",
                   "%+18.15f",
                   "%+18.15f",
                   "%+18.15f",
                   "%+18.15f",
                   "%+18.15f",]
        np.savetxt(OutputFile, result[1,:], header=headerStr,
                   fmt=fmtList)
end
