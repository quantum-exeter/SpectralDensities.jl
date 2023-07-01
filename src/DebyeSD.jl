"""
    struct DebyeSD <: AbstractSD

DebyeSD represents a Debye spectral density.
It is characterized by an amplitude `α` representing the strength of the coupling
and the cutoff frequency `ωc`. That is
```math
J(\\omega) = \\frac{2\\alpha}{\\pi}\\frac{\\omega\\omega_c^2}{\\omega^2 + \\omega_c^2}
```

# Fields
- `α::Float64`: The amplitude `α`, indicating the strength of the coupling.
- `ωc::Float64`: The cutoff frequency.

"""
struct DebyeSD <: AbstractSD
    α::Float64
    ωc::Float64
end

"""
    DebyeSD(α, ωc)

Construct a Debye spectral density with the given amplitude `α` and cutoff frequency `ωc`.

# Arguments
- `α`: The amplitude `α`, indicating the strength of the coupling.
- `ωc`: The cutoff frequency.

# Returns
- An instance of the `DebyeSD` struct representing the Debye spectral density.

"""
DebyeSD(α, ωc) = DebyeSD(float(α), float(ωc))

sdoverω(J::DebyeSD, ω) = (2*J.α/π)*J.ωc^2/(J.ωc^2 + ω^2)

reorganisation_energy(J::DebyeSD) = J.α*J.ωc

"""
    struct OverdampedSD <: AbstractSD

Represents an overdamped spectral density. This is just an alias for `DebyeSD`.

"""
const OverdampedSD = DebyeSD

"""
    struct LorentzDrudeSD <: AbstractSD

Represents a Lorentz-Drude spectral density. This is just an alias for `DebyeSD`.

"""
const LorentzDrudeSD = DebyeSD