"""
    struct InversePolyKernelSD <: AbstractSD

InversePolyKernelSD represents a spectral density whose corresponding
memory kernel in frequency space is given by one over a polynomial in frequency.
It is characterized by the degree of the polynomial `deg` and the list of
coefficients `coefs` (in ascending order of powers).
That is,
```math
J(\\omega) = \\frac{1}{\\pi}\\mathrm{Im}\\left[\\frac{1}{\\sum_{k=0}^{\\mathrm{deg}}\\mathrm{coeffs}[k]\\omega^k}\\right]
```

# Fields
- `deg::Int64`: The degree of the polynomial.
- `coeffs::AbstractVector{ComplexF64}`: Coefficients of the polynomial (in ascending order of powers). The length of the vector should be `deg + 1`.

"""
struct InversePolyKernelSD <: AbstractSD
    deg::Int64
    coeffs::AbstractVector{ComplexF64}
end

"""
    InversePolyKernelSD(coeffs::AbstractVector)

Construct a InversePolyKernelSD spectral density with the given coefficients.

# Arguments
- `coeffs::AbstractVector`: A vector containing the coefficients of the polynomial (in ascending order of powers).

# Returns
- An instance of the InversePolyKernelSD struct representing the spectral density.

"""
InversePolyKernelSD(coeffs::AbstractVector) = InversePolyKernelSD(length(coeffs)-1, ComplexF64.(coeffs))

sd(J::InversePolyKernelSD, ω) = imag_memory_kernel_ft(J, ω)/π

imag_memory_kernel_ft(J::InversePolyKernelSD, ω) = imag(memory_kernel_ft(J, ω))

real_memory_kernel_ft(J::InversePolyKernelSD, ω) = real(memory_kernel_ft(J, ω))

memory_kernel_ft(J::InversePolyKernelSD, ω) = inv(evalpoly(ω, J.coeffs))