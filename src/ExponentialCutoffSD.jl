"""
    struct ExponentialCutoffSD{T <: AbstractSD} <: AbstractSD

ExponentialCutoffSD represents a spectral density with an exponential frequency cutoff.
It is parameterized by the underlying spectral density `J`,
and the frequency cutoff `ωcutoff`.
That is
```math
J_\\mathrm{exp}(\\omega) = J(\\omega)e^{-\\omega/\\omega_c}
```

# Fields
- `J::T`: The underlying spectral density.
- `ωcutoff::Float64`: The frequency cutoff.

"""
struct ExponentialCutoffSD{T <: AbstractSD} <: AbstractSD
    J::T
    ωcutoff::Float64
end

"""
    ExponentialCutoffSD(J::AbstractSD, ωcutoff)

Construct a spectral density with an exponential frequency cutoff using
the underlying spectral density `J` and the given frequency cutoff `ωcutoff`.

# Arguments
- `J::AbstractSD`: The underlying spectral density.
- `ωcutoff`: The frequency cutoff.

# Returns
- An instance of the `ExponentialCutoffSD` struct representing the spectral density with an exponential frequency cutoff.

"""
ExponentialCutoffSD(J, ωcutoff) = ExponentialCutoffSD(J, float(ωcutoff))

sdoverω(J::ExponentialCutoffSD, ω) = sdoverω(J.J, ω)*exp(-abs(ω)/J.ωcutoff)

reorganisation_energy(J::ExponentialCutoffSD{OhmicSD}) = J.J.α*J.ωcutoff
reorganisation_energy(J::ExponentialCutoffSD{PolySD}) = J.J.α*J.ωcutoff^J.J.n*factorial(J.J.n-1)

frequency_cutoff(J::ExponentialCutoffSD; tol=eps()) = -J.ωcutoff*log(tol)