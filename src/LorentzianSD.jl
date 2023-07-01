"""
    struct LorentzianSD <: AbstractSD

LorentzianSD represents a Lorentzian spectral density.
It is characterized by an amplitude `α` representing the strength of the coupling,
the peak centre frequency `ω0`, and the width of the Lorentzian peak `Γ`.
 That is
```math
J(\\omega) = \\frac{\\alpha\\Gamma}{\\pi}\\frac{\\omega}{(\\omega^2 - \\omega_0^2)^2 + \\omega^2\\Gamma^2}
```

# Fields
- `α::Float64`: The amplitude `α`, indicating the strength of the coupling.
- `ω0::Float64`: The centre frequency of the Lorentzian peak.
- `Γ::Float64`: The width of the Lorentzian peak.

"""
struct LorentzianSD <: AbstractSD
    α::Float64
    ω0::Float64
    Γ::Float64
end

"""
    LorentzianSD(α, ω0, Γ)

Construct a Lorentzian spectral density with the given amplitude `α`, centre frequency `ω0`, and width `Γ`.

# Arguments
- `α`: The amplitude `α`, indicating the strength of the coupling.
- `ω0`: The centre frequency of the Lorentzian peak.
- `Γ`: The width of the Lorentzian peak.

# Returns
- An instance of the `LorentzianSD` struct representing the Lorentzian spectral density.

"""
LorentzianSD(α, ω0, Γ) = LorentzianSD(float(α), float(ω0), float(Γ))

sdoverω(J::LorentzianSD, ω) = (J.α*J.Γ/π)/((ω^2 - J.ω0^2)^2 + (J.Γ*ω)^2)

reorganisation_energy(J::LorentzianSD) = J.α/J.ω0^2/2

"""
    struct UnderdampedSD <: AbstractSD

Represents an underdamped spectral density. This is just an alias for `LorentzianSD`.

"""
const UnderdampedSD = LorentzianSD