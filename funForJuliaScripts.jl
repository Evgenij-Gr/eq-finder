module FunForJulia
export reducedSystem, reducedSystemJac, getLyapunovData, getPtOnAttr, getClassOfSymm, getLyapunovVal, prepareData

using PyCall
using DynamicalSystems
using DifferentialEquations
using Combinatorics
using Distances
using NumericalIntegration
using LinearAlgebra

np = pyimport("numpy")

struct FourBiharmonicPhaseOscillators
    paramW
    paramA
    paramB
    paramR
end

function funG(syst,fi)
    -sin(fi + syst.paramA) + syst.paramR * sin(2 * fi + syst.paramB)
end

function getFullSystem(syst, phis)
    rhsPhis = [0.,0.,0.,0.]
    for i = 1:length(rhsPhis)
        elem = syst.paramW
        for j = 1:length(rhsPhis)
            elem += 0.25 * funG(syst,phis[i]-phis[j])
        end
        rhsPhis[i] = elem
    end
    return rhsPhis
end

function getReducedSystem(syst, gammas)
    phis = append!([0.0], gammas)
    rhsPhi = getFullSystem(syst,phis)
    rhsGamma = [0.,0.,0.,0.]
    for i in 1:length(rhsGamma)
        rhsGamma[i] = rhsPhi[i]-rhsPhi[1]
    end
    return rhsGamma
end

function NotDiagComponentJac(syst, x, y)
    return ((cos(x - y + syst.paramA) - 2 * syst.paramR * cos(2 * (x - y) + syst.paramB)) -
            (cos((-y) + syst.paramA) - 2 * syst.paramR * cos(2 * (-y) + syst.paramB))) / 4
end

function DiagComponentJac3d(syst, x, y, z)
    return ((-cos(x + syst.paramA) + 2 * syst.paramR * cos(2 * x + syst.paramB)) +
        (-cos(x - y + syst.paramA) + 2 * syst.paramR * cos(2 * (x - y) + syst.paramB)) +
        (-cos(x - z + syst.paramA) + 2 * syst.paramR * cos(2 * (x - z) + syst.paramB)) -
        (cos((-x) + syst.paramA) - 2 * syst.paramR * cos(2 * (-x) + syst.paramB))) / 4

end

function getReducedSystemJac(syst, X)
    x, y, z = X
    return [DiagComponentJac3d(syst, x, y, z) NotDiagComponentJac(syst, x, y) NotDiagComponentJac(syst, x, z);
        NotDiagComponentJac(syst, y, x) DiagComponentJac3d(syst, y, x, z) NotDiagComponentJac(syst, y, z);
        NotDiagComponentJac(syst, z, x) NotDiagComponentJac(syst, z, y) DiagComponentJac3d(syst, z, x, y)]
end

function fullSystem(du, u, p, t)
    w = p[1]; α = p[2]; β = p[3]; ρ = p[4]
    rhs  = FourBiharmonicPhaseOscillators(w, α, β, ρ)
    temp = getFullSystem(rhs, u)
    du[1] = temp[1]
    du[2] = temp[2]
    du[3] = temp[3]
    du[4] = temp[4]
end

@inline @inbounds function reducedSystem(u, p, t)
    w = p[1]; α = p[2]; β = p[3]; ρ = p[4]
    rhs = FourBiharmonicPhaseOscillators(w, α, β, ρ)
    temp = getReducedSystem(rhs, u)
    return SVector{3}(temp[2], temp[3], temp[4])
end

@inline @inbounds function reducedSystemJac(u, p, t)
    w,α,β,ρ = p
    rhs = FourBiharmonicPhaseOscillators(w,α,β,ρ)
    return getReducedSystemJac(rhs, u)
end


function getLyapunovVal(params)
    i, j, a, b, r, startPtX, startPtY, startPtZ, rhsInfNormVal = params
    if (rhsInfNormVal > 1e-7)
        a4d = ContinuousDynamicalSystem(reducedSystem, rand(3), [0.5,a,b,r], reducedSystemJac)
        λ = lyapunov(a4d, 100000.0, u0 = [startPtX, startPtY, startPtZ], dt = 0.1, Ttr = 10.0)
    else
        λ = -1
    end
    return[i, j, a, b, r, λ]
end

function getSolFullSyst(params, startPt,  maxTime, evalTs = 0, tol = 1e-13)
    u0 = vcat([0], startPt)
    tspan = (0.0, maxTime)
    prob = ODEProblem(fullSystem, u0, tspan, params)
    alg = DP8()
    if (evalTs != 0)
        sol = solve(prob, alg, saveat = evalTs, reltol = tol, abstol = tol)
    else
        sol = solve(prob, alg, reltol = tol, abstol = tol)
    end
end

function detective_fun2(X)
    x1 = X[1]
    x2 = X[2]
    x3 = X[3]
    x4 = X[4]
    res = zeros(0)
    for i = 1:size(x1)[1]
        append!(res, sin(3.0*x2[i]) * sin(6.0*x3[i]) * sin(9.0*x4[i]))
    end
    return res
end

function polynomial_detectiveSimp(X, ts)
    permutList = collect(permutations([1, 2, 3, 4]))
    res = zeros(0)
    for i = 1:24
        a = permutList[i]
        curTrajCoords = [X[a[1], :], X[a[2], :], X[a[3], :], X[a[4], :]]
        obsArray = detective_fun2(curTrajCoords)
        obsAvg = integrate(ts, obsArray, SimpsonEven())/ts[end]
        append!(res, obsAvg)
    end
    return res
end

function classifySymmetry(distMat)
    el1 = distMat[2][1]
    el2 = distMat[3][1]
    el3 = distMat[4][1]
    if (log10(el1) < -1.25)
        symType = "T1"
    elseif (log10(el1) > -1.25 && log10(el2) < -1.9 && log10(el3) > -1.25)
        symType = "T2"
    elseif (log10(el1) > -1.25 && log10(el2) > -1.9 && log10(el3) > -1.25)
        symType = "T0"
    else
        symType = "unknown $el1 , $el2"
    end
    return symType
end

function getSymmType(params, startPt,  maxT = 20000, evalTs = 0.1)
    sol = getSolFullSyst(params, startPt,  maxT, evalTs)
    pts = np.array(sol.u)
    permutList = [(1, 2, 3, 4), (2, 3, 4, 1), (3, 4, 1, 2), (4, 1, 2, 3)]
    kAs = zeros(24,0)

    for p in permutList
        newTraj = np.array([pts[:, p[1]], pts[:, p[2]], pts[:, p[3]], pts[:, p[4]]])
        kA = polynomial_detectiveSimp(newTraj, sol.t)
        kAs = hcat(kAs, kA)
    end

    distMat = pairwise(Euclidean(), kAs, kAs)
    return classifySymmetry(distMat)
end

function getClassOfSymm(params)
    i, j, a, b, r, startPtX, startPtY, startPtZ, rhsInfNormVal = params
    if (rhsInfNormVal > 1e-7)
        T = getSymmType([0.5, a, b, r], [startPtX, startPtY, startPtZ])
    else
        T = "noAttr"
    end
    return["$i", "$j", "$a", "$b", "$r", T]
end

function getPtOnAttr(params)
    i, j, a, b, r, startPtX, startPtY, startPtZ, maxTime, evalTs = params
    sol = getSolFullSyst([0.5, a, b, r], [startPtX, startPtY, startPtZ], maxTime, evalTs)
    lastPt = [sol.u[end][2] - sol.u[end][1], sol.u[end][3] - sol.u[end][1], sol.u[end][4] - sol.u[end][1]]
    a4d = FourBiharmonicPhaseOscillators(0.5, a, b, r)
    return[i, j, a, b, r, lastPt[1], lastPt[2], lastPt[3], norm(getReducedSystem(a4d, lastPt),Inf)]
end

function prepareData(dataToPrep)
    data = Vector[dataToPrep[1,:]]
    for i in 2:size(dataToPrep[:,1])[1]
                    data=hcat(data,[dataToPrep[i,:]])
    end
    data
end

function getLyapunovData(params)
    i, j, a, b, r, startPtX, startPtY, startPtZ, rhsInfNormVal = params
    if (rhsInfNormVal > 1e-7)
        a4d = ContinuousDynamicalSystem(reducedSystem, rand(3), [0.5,a,b,r], reducedSystemJac)
        λλ = lyapunovspectrum(a4d, 100000.0, u0 = [startPtX, startPtY, startPtZ], dt = 0.1, Ttr = 10.0)
    else
        λλ = [-1, -1, -1]
    end
    return[i, j, a, b, r, λλ[1], λλ[2], λλ[3]]
end
end
