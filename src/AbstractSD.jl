"""
    abstract type AbstractSD

AbstractSD represents an abstract bath spectral density.

Any subtype of `AbstractSD` must at least define either `sd` or `sdoverω` for the new type.

"""
abstract type AbstractSD end

(J::AbstractSD)(ω) = sd(J, ω)

Base.length(::AbstractSD) = 1
Base.iterate(J::AbstractSD) = (J, nothing)
Base.iterate(::AbstractSD, ::Any) = nothing
Base.isempty(::AbstractSD) = false

"""
    sd(J::T, ω) where T <: AbstractSD

Evaluate the spectral density represented by `J` at a given frequency `ω`, i.e. `J(ω)`.

# Arguments
- `J::T`: The spectral density.
- `ω`: The frequency at which the spectral density is evaluated.

# Returns
- The spectral density `J` at the frequency `ω`.

"""
sd(J::T, ω) where T <: AbstractSD = sdoverω(J, ω)*ω

"""
    sdoverω(J::T, ω) where T <: AbstractSD

Evaluate the spectral density represented by `J` divided by a given frequency `ω`, i.e. `J(ω)/ω`.

# Arguments
- `J::T`: The spectral density.
- `ω`: The frequency at which the spectral density is evaluated.

# Returns
- The spectral density `J(ω)` divided by `ω`.

"""
sdoverω(J::T, ω) where T <: AbstractSD = sd(J, ω)/ω

"""
    reorganisation_energy(J::AbstractSD)

Calculate the reorganization energy of a given spectral density `J`, i.e.
```math
\\int_0^\\infty \\frac{J(\\omega)}{\\omega} \\mathrm{d}\\omega 
```

# Arguments
- `J::AbstractSD`: The spectral density.

# Returns
- The reorganization energy of the spectral density `J`.

"""
reorganisation_energy(J::AbstractSD) = quadgk(ω -> sdoverω(J,ω), 0.0, Inf)[1]

"""
    correlations(J::AbstractSD, τ, β)

Calculate the correlation function for a spectral density `J` at a given time
delay `τ` and inverse temperature `β`, that is
```math
\\mathcal{C}(\\tau) =
\\int_0^\\infty J(\\omega)\\coth\\left(\\frac{\\hbar\\omega\\beta}{2}\\right)\\cos(\\omega\\tau)\\mathrm{d}\\omega
-i\\int_0^\\infty J(\\omega)\\sin(\\omega\\tau)\\mathrm{d}\\omega.
```

# Arguments
- `J::AbstractSD`: The spectral density.
- `τ`: The time delay at which the correlation function is calculated.
- `β`: The inverse temperature.

# Returns
- The correlation function for the spectral density `J` at the given time delay `τ` and inverse temperature β.

"""
function correlations(J::AbstractSD, τ, β; ωcutoff=Inf)
    return correlations_real(J, τ, β; ωcutoff=ωcutoff) + im*correlations_imag(J, τ; ωcutoff=ωcutoff)
end

"""
    correlations_real(J::AbstractSD, τ, β)

Calculate the real part of the correlation function for a spectral density `J`
at a given time delay `τ` and inverse temperature `β`, that is
```math
\\mathrm{Re}[\\mathcal{C}(\\tau)] = \\int_0^\\infty J(\\omega)\\coth\\left(\\frac{\\hbar\\omega\\beta}{2}\\right)\\cos(\\omega\\tau)\\mathrm{d}\\omega.
```

# Arguments
- `J::AbstractSD`: The spectral density.
- `τ`: The time delay at which the correlation function is calculated.
- `β`: The inverse temperature.

# Returns
- The real part of the correlation function for the spectral density `J` at the
given time delay `τ` and inverse temperature `β`.

"""
function correlations_real(J::AbstractSD, τ, β; ωcutoff=Inf)
    IRe = quadgk(ω -> sd(J,ω)*cos(ω*τ)*coth(ω*β/2), zero(τ), ωcutoff)
    return IRe[1]
end

"""
    correlations_imag(J::AbstractSD, τ)

Calculate the imaginary part of the correlation function for a spectral density
`J` at a given time delay `τ`, that is
```math
\\mathrm{Im}[\\mathcal{C}(\\tau)] = -\\int_0^\\infty J(\\omega)\\sin(\\omega\\tau)\\mathrm{d}\\omega.
```

# Arguments
- `J::AbstractSD`: The spectral density.
- `τ`: The time delay at which the correlation function is calculated.

# Returns
- The imaginary part of the correlation function for the spectral density `J` at
the given time delay `τ`.

"""
function correlations_imag(J::AbstractSD, τ; ωcutoff=Inf)
    IIm = quadgk(ω -> -sd(J,ω)*sin(ω*τ), zero(τ), ωcutoff)
    return IIm[1]
end

"""
    memory_kernel(J::AbstractSD, τ)

Calculate the memory kernel for a spectral density `J` at a given time delay `τ`,
that is
```math
\\mathcal{K}(\\tau) = 2\\Theta(\\tau)\\int_0^\\infty J(\\omega)\\sin(\\omega\\tau)\\mathrm{d}\\omega.
```
where ``\\Theta`` is the Heavisde theta function.

# Arguments
- `J::AbstractSD`: The spectral density.
- `τ`: The time delay at which the memory kernel is evaluated.
- `ωcutoff`: (optinal, default: Inf) Frequency cutoff to be used when calculating the correlation function.

# Returns
- The memory kernel for the spectral density `J` at the given time delay `τ`.

"""
memory_kernel(J::AbstractSD, τ; ωcutoff=Inf) = τ <= zero(τ) ? zero(τ) : -2*correlations_imag(J, τ; ωcutoff=ωcutoff)

"""
    imag_memory_kernel_ft(J::AbstractSD, ω)

Calculate the imaginary part of the Fourier-transform of the  memory kernel for a
spectral density `J` at a given frequency `ω`.

# Arguments
- `J::AbstractSD`: The spectral density.
- `ω`: The frequency at which the imaginary part of the Fourier-transform of the memory kernel is evaluated.

# Returns
- The imaginary part of the Fourier-transform of the memory kernel for the spectral density `J` at the given frequency `ω`.

"""
imag_memory_kernel_ft(J::AbstractSD, ω) = π*J(ω)

"""
    real_memory_kernel_ft(J::AbstractSD, ω)

Calculate the real part of the Fourier-transform of the  memory kernel for a
spectral density `J` at a given frequency `ω`.

# Arguments
- `J::AbstractSD`: The spectral density.
- `ω`: The frequency at which the real part of the Fourier-transform of the memory kernel is evaluated.

# Returns
- The real part of the Fourier-transform of the memory kernel for the spectral density `J` at the given frequency `ω`.

"""
real_memory_kernel_ft(J::AbstractSD, ω) = SingularIntegrals.kramers_kronig(ω -> imag_memory_kernel_ft(J,ω), ω; cutoff=frequency_cutoff(J))

"""
    memory_kernel_ft(J::AbstractSD, ω)

Calculate the Fourier-transform of the  memory kernel for a
spectral density `J` at a given frequency `ω`.

# Arguments
- `J::AbstractSD`: The spectral density.
- `ω`: The frequency at which the Fourier-transform of the memory kernel is evaluated.

# Returns
- The Fourier-transform of the memory kernel for the spectral density `J` at the given frequency `ω`.

"""
memory_kernel_ft(J::AbstractSD, ω) = real_memory_kernel_ft(J, ω) + 1im*imag_memory_kernel_ft(J, ω)

"""
    frequency_cutoff(J::AbstractSD; tol=eps())

Return the frequency cutoff of the spectral density `J` to accuracy set
by the tolerance parameter `tol`.

# Arguments
- `J::AbstractSD`: The spectral density.
- `tol`: The tolerance at which to truncate the spectral density.

# Returns
- The frequency cutoff for the desired tolerance.

"""
frequency_cutoff(J::AbstractSD; tol=eps()) = typemax(tol)

"""
    frequency_step(J::AbstractSD; tol=eps())

Calculate an appropriate frequency step for accurate numerical
discretisation of the spectral density `J`.

# Arguments
- `J::AbstractSD`: The spectral density.
- `tol::Real`: (optional, default=eps()) The tolerance used to determine the appropriate frequency setp.

# Returns
- The calculated frequency step for discretisation at the given tolerance.
"""
frequency_step(J::AbstractSD; tol=eps()) = eps()