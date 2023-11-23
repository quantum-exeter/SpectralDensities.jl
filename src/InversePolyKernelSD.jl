"""
    struct InversePolyKernelSD <: AbstractSD

InversePolyKernelSD represents a spectral density whose corresponding
memory kernel in frequency space is given by one over a polynomial in frequency.
It is characterized by the degree of the polynomial `deg` and the list of
coefficients `coefs` (in ascending order of powers).
That is,
```math
J(\\omega) = \\frac{1}{\\pi}\\mathrm{Im}\\left[\\frac{1}{\\sum_{k=0}^{\\mathrm{deg}}I_k\\mathrm{coeffs}[k+1]\\omega^k}\\right],
```
where ``math I_k = 1`` if `k` is even, and ``math I_k = i`` if `k` is odd.

# Fields
- `deg::Int64`: The degree of the polynomial.
- `coeffs::AbstractVector{ComplexF64}`: Coefficients of the polynomial (in ascending order of powers). The length of the vector should be `deg + 1`.

"""
struct InversePolyKernelSD <: AbstractSD
    deg::Int64
    coeffs::AbstractVector{Float64}
    real::AbstractVector{Float64}
    imag::AbstractVector{Float64}
end

"""
    InversePolyKernelSD(coeffs::AbstractVector)

Construct a InversePolyKernelSD spectral density with the given coefficients.

# Arguments
- `coeffs::AbstractVector`: A vector containing the coefficients of the polynomial (in ascending order of powers).

# Returns
- An instance of the InversePolyKernelSD struct representing the spectral density.

"""
function InversePolyKernelSD(coeffs::Vector{<:Real})
    re = zeros(length(coeffs))
    im = zeros(length(coeffs))
    for k in 1:length(coeffs)
        if isodd(k)
            re[k] = coeffs[k]
        else
            im[k] = coeffs[k]
        end
    end
    return InversePolyKernelSD(length(coeffs)-1, Float64.(coeffs), re, im)
end

function InversePolyKernelSD(coeffs::Vector{<:Complex})
    re = zeros(length(coeffs))
    im = zeros(length(coeffs))
    merged = zeros(length(coeffs))
    for k in 1:length(coeffs)
        if isodd(k)
            re[k] = real(coeffs[k])
            merged[k] = re[k]
        else
            im[k] = imag(coeffs[k])
            merged[k] = im[k]
        end
    end
    return InversePolyKernelSD(length(coeffs)-1, merged, re, im)
end

function InversePolyKernelSD(real::Vector{<:Real}, imag::Vector{<:Real})
    len = length(real) + length(imag)
    re = zeros(len)
    im = zeros(len)
    merged = zeros(len)
    for k in 1:len
        if isodd(k)
            re[k] = real[1 + k ÷ 2]
            merged[k] = re[k]
        else
            im[k] = imag[k ÷ 2]
            merged[k] = im[k]
        end
    end
    return InversePolyKernelSD(len-1, merged, re, im)
end

"""
    InversePolyKernelSD(J::LorentzianSD)

Construct a InversePolyKernelSD spectral density from a Lorentzian spectral density.

# Arguments
- `J::LorentzianSD`: The Lorentzian spectral density.

# Returns
- An instance of the InversePolyKernelSD that represents the Lorentzian spectral density.

"""
InversePolyKernelSD(J::LorentzianSD) = InversePolyKernelSD([J.ω0^2/J.α, -J.Γ/J.α, -1/J.α])

Base.convert(::Type{InversePolyKernelSD}, J::LorentzianSD) = InversePolyKernelSD(J)

"""
    InversePolyKernelSD(J::DebyeSD)

Construct a InversePolyKernelSD spectral density from a Debye spectral density.

# Arguments
- `J::DebyeSD`: The Debye spectral density.

# Returns
- An instance of the InversePolyKernelSD that represents the Debye spectral density.

"""
InversePolyKernelSD(J::DebyeSD) = InversePolyKernelSD([1/(2*J.α*J.ωc), -1/(2*J.α*J.ωc^2)])

Base.convert(::Type{InversePolyKernelSD}, J::DebyeSD) = InversePolyKernelSD(J)

sd(J::InversePolyKernelSD, ω) = imag_memory_kernel_ft(J, ω)/π

imag_memory_kernel_ft(J::InversePolyKernelSD, ω) = imag(memory_kernel_ft(J, ω))

real_memory_kernel_ft(J::InversePolyKernelSD, ω) = real(memory_kernel_ft(J, ω))

memory_kernel_ft(J::InversePolyKernelSD, ω) = inv(evalpoly(ω, J.real) + 1im*evalpoly(ω, J.imag))