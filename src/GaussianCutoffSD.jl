"""
    struct GaussianCutoffSD{T <: AbstractSD} <: AbstractSD

GaussianCutoffSD represents a spectral density with a Gaussian frequency cutoff.
It is parameterized by the underlying spectral density `J`,
and the frequency cutoff `ωcutoff`.
That is
```math
J_\\mathrm{gauss}(\\omega) = J(\\omega)e^{-\\omega^2/\\omega_c^2}
```

# Fields
- `J::T`: The underlying spectral density.
- `ωcutoff::Float64`: The frequency cutoff.

"""
struct GaussianCutoffSD{T <: AbstractSD} <: AbstractSD
    J::T
    ωcutoff::Float64
end

"""
    GaussianCutoffSD(J::AbstractSD, ωcutoff)

Construct a spectral density with a Gaussian frequency cutoff using
the underlying spectral density `J` and the given frequency cutoff `ωcutoff`.

# Arguments
- `J::AbstractSD`: The underlying spectral density.
- `ωcutoff`: The frequency cutoff.

# Returns
- An instance of the `GaussianCutoffSD` struct representing the spectral density with a Gaussian frequency cutoff.

"""
GaussianCutoffSD(J, ωcutoff) = GaussianCutoffSD(J, float(ωcutoff))

sdoverω(J::GaussianCutoffSD, ω) = sdoverω(J.J, ω)*exp(-(ω/J.ωcutoff)^2)

reorganisation_energy(J::GaussianCutoffSD{OhmicSD}) = J.J.α*J.ωcutoff*sqrt(π)/2
reorganisation_energy(J::GaussianCutoffSD{PolySD}) = J.J.α*J.ωcutoff^J.J.n*gamma(J.J.n/2)/2