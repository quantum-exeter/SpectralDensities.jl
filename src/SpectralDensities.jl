module SpectralDensities

using QuadGK

include("AbstractSD.jl")

export AbstractSD, sd, sdoverω, reorganisation_energy, correlations

end