"""
    struct HardCutoffSD{T <: AbstractSD} <: AbstractSD

HardCutoffSD represents a spectral density with a hard frequency cutoff.
It is parameterized by the underlying spectral density `J`,
and the frequency cutoff `ωcutoff`.
That is
```math
J_\\mathrm{hard}(\\omega) = J(\\omega)\\Theta(\\omega_c - \\omega)
```
where ``\\Theta`` is the Heaviside theta function.

# Fields
- `J::T`: The underlying spectral density.
- `ωcutoff::Float64`: The frequency cutoff.

"""
struct HardCutoffSD{T <: AbstractSD} <: AbstractSD
    J::T
    ωcutoff::Float64
end

"""
    HardCutoffSD(J::AbstractSD, ωcutoff)

Construct a spectral density with a hard frequency cutoff using the underlying
spectral density `J` and the given frequency cutoff `ωcutoff`.

# Arguments
- `J::AbstractSD`: The underlying spectral density.
- `ωcutoff::Float64`: The frequency cutoff.

# Returns
- An instance of the `HardCutoffSD` struct representing the spectral density with a hard frequency cutoff.

"""
HardCutoffSD(J::AbstractSD, ωcutoff) = HardCutoffSD(J, float(ωcutoff))

sdoverω(J::HardCutoffSD, ω) = abs(ω) > J.ωcutoff ? zero(ω) : sdoverω(J.J, ω)

reorganisation_energy(J::HardCutoffSD) = quadgk(ω -> sdoverω(J.J,ω), 0.0, J.ωcutoff)[1]
reorganisation_energy(J::HardCutoffSD{OhmicSD}) = J.J.α*J.ωcutoff
reorganisation_energy(J::HardCutoffSD{PolySD}) = J.J.α*J.ωcutoff^J.J.n/J.J.n

frequency_cutoff(J::HardCutoffSD; tol=eps()) = J.ωcutoff