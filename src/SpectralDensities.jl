module SpectralDensities

using QuadGK

include("AbstractSD.jl")

export AbstractSD, sd, sdoverω, reorganisation_energy, correlations

include("OhmicSD.jl")

export OhmicSD

include("PolySD.jl")

export PolySD

include("LorentzianSD.jl")

export LorentzianSD

include("DebyeSD.jl")

export DebyeSD, OverdampedSD, LorentzDrudeSD

end