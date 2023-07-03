module SpectralDensities

using QuadGK
using SpecialFunctions

include("AbstractSD.jl")

export AbstractSD, sd, sdoverω, reorganisation_energy, correlations

include("OhmicSD.jl")

export OhmicSD

include("PolySD.jl")

export PolySD

include("LorentzianSD.jl")

export LorentzianSD, UnderdampedSD

include("DebyeSD.jl")

export DebyeSD, OverdampedSD, LorentzDrudeSD

include("HardCutoffSD.jl")

export HardCutoffSD

include("ExponentialCutoffSD.jl")

export ExponentialCutoffSD

include("GaussianCutoffSD.jl")

export GaussianCutoffSD

module WeakCoupling
    using ForwardDiff
    using QuadGK
    using ..SpectralDensities

    include("WeakCoupling.jl")

    export weak_coupling_Δ, weak_coupling_Δprime,
           weak_coupling_Σ, weak_coupling_Σprime,
           cauchy_quadgk, hadamard_quadgk
end

export WeakCoupling

end