using Distributed
# Usage: julia LyapVals.jl <pathToDataFile> <outputMask> <outputDir>
addprocs(8)
@everywhere using PyCall
@everywhere np =  pyimport("numpy")
@everywhere heteroclinicsData = np.loadtxt($(ARGS[1]))
@everywhere function prepareData(dataToPrep)
    data = Vector[dataToPrep[1,:]]
    for i in 2:size(dataToPrep[:,1])[1]
                    data=hcat(data,[dataToPrep[i,:]])
    end
    data
end
@everywhere data = prepareData(heteroclinicsData)
@everywhere include("funForJuliaScripts.jl")
@everywhere using Main.FunForJulia

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
