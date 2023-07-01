module SpectralDensities

using QuadGK

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

end