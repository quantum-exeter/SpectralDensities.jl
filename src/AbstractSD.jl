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

Calculate the correlation function for a spectral density `J` at a given time delay `τ` and inverse temperature `β`.

# Arguments
- `J::AbstractSD`: The spectral density.
- `τ`: The time delay at which the correlation function is calculated.
- `β`: The inverse temperature.

# Returns
- The correlation function for the spectral density `J` at the given time delay `τ` and inverse temperature β.

"""
function correlations(J::AbstractSD, τ, β)
    IntRe(ω) = sd(J,ω)*cos(ω*τ)*tanh(ω*β/2)
    IntIm(ω) = -sd(J,ω)*sin(ω*τ)
    IRe = quadgk(IntRe, 0.0, Inf)[1]
    IIm = quadgk(IntIm, 0.0, Inf)[1]
    return IRe + 1im*IIm
end

"""
    memory_kernel(J::AbstractSD, τ)

Calculate the memory kernel for a spectral density `J` at a given time delay `τ`,
that is
```math
\\mathcal{K}(\\tau) = 2\\Theta(\\tau)\\int_0^\\infty J(\\omega)\\sin(\\omega)\\mathrm{d}\\omega,
```
where ``\\Theta`` is the Heavisde theta function.

# Arguments
- `J::AbstractSD`: The spectral density.
- `τ`: The time delay at which the memory kernel is evaluated.

# Returns
- The memory kernel for the spectral density `J` at the given time delay `τ`.

"""
memory_kernel(J::AbstractSD, τ) = τ <= zero(τ) ? zero(τ) : quadgk(ω -> 2*J(ω)*sin(ω*τ), zero(τ), Inf)[1]

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