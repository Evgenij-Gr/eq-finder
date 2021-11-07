using Distributed
# Usage: julia classificationSymmetry.jl <pathToDataFile> <outputMask> <outputDir>
addprocs(8)
@everywhere using PyCall
@everywhere np =  pyimport("numpy")
@everywhere attrData = np.loadtxt($(ARGS[1]))
@everywhere function prepareData(dataToPrep)
    data = Vector[dataToPrep[1,:]]
    for i in 2:size(dataToPrep[:,1])[1]
                    data=hcat(data,[dataToPrep[i,:]])
    end
    data
end
@everywhere data = prepareData(attrData)
@everywhere include("funForJuliaScripts.jl")
@everywhere using Main.FunForJulia

using Dates
timeOfRun = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
nameOutputFile = ARGS[2]
pathToOutputDir = ARGS[3]
outputFileMask = string(nameOutputFile, timeOfRun)
outputFileMask = string(outputFileMask, ".txt")
OutputFile = string(pathToOutputDir, outputFileMask)
@time begin
    result = pmap(getClassOfSymm, data)
end

if !isempty(result)
        headerStr = (
                "i  j  alpha  beta  r  ClassOfSymmetry\n0  1  2      3     4  5")
        fmtList = ["%s",
                   "%s",
                   "%18s",
                   "%18s",
                   "%s",
                   "%s",]
        np.savetxt(OutputFile, result[1,:], header=headerStr,
                   fmt=fmtList)
end