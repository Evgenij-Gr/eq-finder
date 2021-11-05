module ReducedSysLyapVals
export reducedSystem, getLyapunovData

import Pkg

Pkg.add("PyCall")
using PyCall

Pkg.add("DynamicalSystems")
using DynamicalSystems

pushfirst!(PyVector(pyimport("sys")."path"), "")
so =  pyimport("SystOsscills")

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


@inline @inbounds function reducedSystem(u, p, t)
    w = p[1]; α = p[2]; β = p[3]; ρ = p[4]
    rhs = FourBiharmonicPhaseOscillators(w, α, β, ρ)
    temp = getReducedSystem(rhs, u)
    return SVector{3}(temp[2], temp[3], temp[4])
end

@inline @inbounds function reducedSystemJac(u, p, t)
    w,α,β,ρ = p
    rhs = FourBiharmonicPhaseOscillators(w,α,β,ρ)
    return @SMatrix [getReducedSystemJac(rhs, u)]
end


function getLyapunovData(params)
    i, j, a, b, r, startPtX, startPtY, startPtZ = params
    a4d = ContinuousDynamicalSystem(reducedSystem, rand(3), [0.5,a,b,r], reducedSystemJac)
    λ = lyapunov(a4d, 100000.0, u0 = [startPtX, startPtY, startPtZ], dt = 0.1, Ttr = 10.0)
    return[i, j, a, b, r, λ]
end
end
