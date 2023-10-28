module SpectralDensities

using QuadGK
using SpecialFunctions

include("AbstractSD.jl")

export AbstractSD, sd, sdoverω, reorganisation_energy, correlations, memory_kernel, imag_memory_kernel_ft, frequency_cutoff

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

include("InversePolyKernelSD.jl")

export InversePolyKernelSD

module SingularIntegrals
    using QuadGK

    include("SingularIntegrals.jl")

    export cauchy_quadgk, hadamard_quadgk
end

export SingularIntegrals

module WeakCoupling
    using ForwardDiff
    using QuadGK
    using ..SpectralDensities
    using ..SingularIntegrals

    include("WeakCoupling.jl")

    export weak_coupling_Δ, weak_coupling_Δprime,
           weak_coupling_Σ, weak_coupling_Σprime
end

export WeakCoupling

end