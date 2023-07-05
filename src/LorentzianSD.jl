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
    LorentzianSD(α::Real, ω0::Real, Γ::Real)

Construct a Lorentzian spectral density with the given amplitude `α`, centre frequency `ω0`, and width `Γ`.

# Arguments
- `α::Real`: The amplitude `α`, indicating the strength of the coupling.
- `ω0::Real`: The centre frequency of the Lorentzian peak.
- `Γ::Real`: The width of the Lorentzian peak.

# Returns
- An instance of the `LorentzianSD` struct representing the Lorentzian spectral density.

"""
LorentzianSD(α::Real, ω0::Real, Γ::Real) = LorentzianSD(float(α), float(ω0), float(Γ))

sdoverω(J::LorentzianSD, ω) = (J.α*J.Γ/π)/((ω^2 - J.ω0^2)^2 + (J.Γ*ω)^2)

reorganisation_energy(J::LorentzianSD) = J.α/J.ω0^2/2

function memory_kernel(J::LorentzianSD, τ)
    if τ <= zero(τ)
        return zero(τ)
    end
    ω1 = sqrt(J.ω0^2 - J.Γ^2/4)
    return J.α*exp(-J.Γ*τ/2)*sin(ω1*τ)/ω1
end

imag_memory_kernel_ft(J::LorentzianSD, ω) = J.α/(J.ω0^2 - ω^2 - 1im*ω*J.Γ)

"""
    struct UnderdampedSD <: AbstractSD

Represents an underdamped spectral density. This is just an alias for `LorentzianSD`.

"""
const UnderdampedSD = LorentzianSD