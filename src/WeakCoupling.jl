"""
    weak_coupling_Δ(J::AbstractSD, ωB, β; ωcutoff=Inf, ħ=one(ωB))

Calculate the weak-coupling coefficient `Δ` for the spectral density `J`,
system Bohr frequency `ωB`, and inverse temperature `β`, defined as
```math
\\Delta_\\beta(\\omega_\\mathrm{B}) = 2\\omega_\\mathrm{B}\\int_0^\\infty
    J(\\omega)\\frac{1}{\\omega^2-\\omega_\\mathrm{B}^2}
    \\coth\\left(\\frac{\\beta\\hbar\\omega}{2}\\right) \\mathrm{d}\\omega
```
See: J.D. Cresser, J. Anders, Phys. Rev. Lett. 127, 250601 (2021).

# Arguments
- `J::AbstractSD`: The spectral density.
- `ωB`: The system Bohr frequency of interest.
- `β`: The inverse temperature.
- `ωcutoff`: (optional, default=Inf) Frequency cutoff to help with the convergence of the integration.
- `ħ`: The value of the reduced Planck constant. Default is 1.

# Returns
- The weak-coupling coefficient `Δ` for the spectral density `J`, system Bohr frequency `ωB`, and inverse temperature `β`.

"""
function weak_coupling_Δ(J::AbstractSD, ωB, β; ωcutoff=Inf, ħ=one(ωB))
    g(ω) = 2*J(ω)*abs(ωB)*coth(β*ħ*ω/2)/(ω + abs(ωB))
    return cauchy_quadgk(g, zero(ωB), ωcutoff, abs(ωB))[1]
end

"""
    weak_coupling_Δprime(J::AbstractSD, ωB, β; ωcutoff=Inf, ħ=one(ωB))

Calculate the weak-coupling coefficient `Δ′` for the spectral density `J`,
system Bohr frequency `ωB`, and inverse temperature `β`, defined as
```math
{\\Delta'}_\\beta(\\omega_\\mathrm{B}) = 2\\int_0^\\infty
    J(\\omega)\\frac{(\\omega^2 + \\omega_\\mathrm{B}^2)}{(\\omega^2-\\omega_\\mathrm{B}^2)^2}
    \\coth\\left(\\frac{\\beta\\hbar\\omega}{2}\\right) \\mathrm{d}\\omega
```
See: J.D. Cresser, J. Anders, Phys. Rev. Lett. 127, 250601 (2021).

Note: The spectral density `J` must support automatic differentiation
with `ForwardDiff.jl`.

# Arguments
- `J::AbstractSD`: The spectral density.
- `ωB`: The system Bohr frequency of interest.
- `β`: The inverse temperature.
- `ωcutoff`: (optional, default=Inf) Frequency cutoff to help with the convergence of the integration.
- `ħ`: The value of the reduced Planck constant. Default is 1.

# Returns
- The weak-coupling coefficient `Δ′` for the spectral density `J`, system Bohr frequency `ωB`, and inverse temperature `β`.

"""
function weak_coupling_Δprime(J::AbstractSD, ωB, β; ωcutoff=Inf, ħ=one(ωB))
    g(ω) = 2*J(ω)*(ω^2 + ωB^2)*coth(β*ħ*ω/2)/(ω + abs(ωB))^2
    g′(ω) = ForwardDiff.derivative(g,ω)
    return hadamard_quadgk(g, g′, zero(ωB), ωcutoff, abs(ωB))[1]
end

"""
    weak_coupling_Σ(J::AbstractSD, ωB; ωcutoff=Inf)

Calculate the weak-coupling coefficient `Σ` for the spectral density `J`
and system Bohr frequency `ωB`, defined as
```math
\\Sigma(\\omega_\\mathrm{B}) = 2\\int_0^\\infty
    J(\\omega)\\frac{\\omega}{\\omega^2-\\omega_\\mathrm{B}^2} \\mathrm{d}\\omega
```
See: J.D. Cresser, J. Anders, Phys. Rev. Lett. 127, 250601 (2021).

# Arguments
- `J::AbstractSD`: The spectral denstiy.
- `ωB`: The system Bohr frequency of interest.
- `ωcutoff`: (optional, default=Inf) Frequency cutoff to help with the convergence of the integration.

# Returns
- The weak-coupling coefficient `Σ` for the spectral density `J` and system Bohr frequency `ωB`.

"""
function weak_coupling_Σ(J::AbstractSD, ωB; ωcutoff=Inf)
    g(ω) = 2*J(ω)*ω/(ω + abs(ωB))
    return cauchy_quadgk(g, zero(ωB), ωcutoff, abs(ωB))[1]
end

"""
    weak_coupling_Σprime(J::AbstractSD, ωB; ωcutoff=Inf)

Calculate the weak-coupling coefficient `Σ′` for the spectral density `J`
and system Bohr frequency `ωB`, defined as
```math
\\Sigma'(\\omega_\\mathrm{B}) = 4\\omega_\\mathrm{B}\\int_0^\\infty
    J(\\omega)\\frac{\\omega}{(\\omega^2-\\omega_\\mathrm{B}^2)^2}
    \\mathrm{d}\\omega
```
See: J.D. Cresser, J. Anders, Phys. Rev. Lett. 127, 250601 (2021).

Note: The spectral density `J` must support automatic differentiation
with `ForwardDiff.jl`.

# Arguments
- `J::AbstractSD`: The spectral density.
- `ωB`: The system Bohr frequency of interest.
- `ωcutoff`: (optional, default=Inf) Frequency cutoff to help with the convergence of the integration.

# Returns
- The weak-coupling coefficient `Σ′` for the spectral density `J` and system Bohr frequency `ωB`.

"""
function weak_coupling_Σprime(J::AbstractSD, ωB; ωcutoff=Inf)
    g(ω) = 4*J(ω)*abs(ωB)*ω/(ω + abs(ωB))^2
    g′(ω) = ForwardDiff.derivative(g,ω)
    return hadamard_quadgk(g, g′, zero(ωB), ωcutoff, abs(ωB))[1]
end