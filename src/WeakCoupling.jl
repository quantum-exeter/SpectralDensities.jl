"""
    weak_coupling_Δ(J::AbstractSD, ωB, β)

Calculate the weak-coupling coefficient `Δ` for the spectral density `J`,
system Bohr frequency `ωB`, and inverse temperature `β`, defined as
```math
\\Delta_\\beta(\\omega_\\mathrm{B}) = 2\\omega_\\mathrm{B}\\int_0^\\infty
    J(\\omega)\\frac{1}{\\omega^2-\\omega_\\mathrm{B}^2}
    \\coth\\left(\\frac{\\beta\\omega}{2}\\right) \\mathrm{d}\\omega
```
See: J.D. Cresser, J. Anders, Phys. Rev. Lett. 127, 250601 (2021).

# Arguments
- `J::AbstractSD`: The spectral density.
- `ωB`: The system Bohr frequency of interest.
- `β`: The inverse temperature.

# Returns
- The weak-coupling coefficient `Δ` for the spectral density `J`, system Bohr frequency `ωB`, and inverse temperature `β`.

"""
function weak_coupling_Δ(J::AbstractSD, ωB, β)
    g(ω) = 2*J(ω)*abs(ωB)*coth(β*ω/2)/(ω + abs(ωB))
    return cauchy_quadgk(g, zero(ωB), Inf, abs(ωB))[1]
end

"""
    weak_coupling_Δprime(J::AbstractSD, ωB, β)

Calculate the weak-coupling coefficient `Δ′` for the spectral density `J`,
system Bohr frequency `ωB`, and inverse temperature `β`, defined as
```math
{\\Delta'}_\\beta(\\omega_\\mathrm{B}) = 2\\int_0^\\infty
    J(\\omega)\\frac{(\\omega^2 + \\omega_\\mathrm{B}^2)}{(\\omega^2-\\omega_\\mathrm{B}^2)^2}
    \\coth\\left(\\frac{\\beta\\omega}{2}\\right) \\mathrm{d}\\omega
```
See: J.D. Cresser, J. Anders, Phys. Rev. Lett. 127, 250601 (2021).

Note: The spectral density `J` must support automatic differentiation
with `ForwardDiff.jl`.

# Arguments
- `J::AbstractSD`: The spectral density.
- `ωB`: The system Bohr frequency of interest.
- `β`: The inverse temperature.

# Returns
- The weak-coupling coefficient `Δ′` for the spectral density `J`, system Bohr frequency `ωB`, and inverse temperature `β`.

"""
function weak_coupling_Δprime(J::AbstractSD, ωB, β)
    g(ω) = 2*J(ω)*(ω^2 + ωB^2)*coth(β*ω/2)/(ω + abs(ωB))^2
    g′(ω) = ForwardDiff.derivative(g,ω)
    return hadamard_quadgk(g, g′, zero(ωB), Inf, abs(ωB))[1]
end

"""
    weak_coupling_Σ(J::AbstractSD, ωB)

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

# Returns
- The weak-coupling coefficient `Σ` for the spectral density `J` and system Bohr frequency `ωB`.

"""
function weak_coupling_Σ(J::AbstractSD, ωB)
    g(ω) = 2*J(ω)*ω/(ω + abs(ωB))
    return cauchy_quadgk(g, zero(ωB), Inf, abs(ωB))[1]
end

"""
    weak_coupling_Σprime(J::AbstractSD, ωB)

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

# Returns
- The weak-coupling coefficient `Σ′` for the spectral density `J` and system Bohr frequency `ωB`.

"""
function weak_coupling_Σprime(J::AbstractSD, ωB)
    g(ω) = 4*J(ω)*abs(ωB)*ω/(ω + abs(ωB))^2
    g′(ω) = ForwardDiff.derivative(g,ω)
    return hadamard_quadgk(g, g′, zero(ωB), Inf, abs(ωB))[1]
end

"""
    cauchy_quadgk(g, a, b, x0=0; kws...)

Computes the Cauchy principal value of the integral of a function `g`
with singularity at `x0` over the interval `[a, b]`, that is
```math
\\mathrm{P.V.}\\int_a^b\\frac{g(x)}{x-x_0}\\mathrm{d}x
```
Note that `x0` must be contained in the interval `[a, b]`.
The actual integration is performed by the `quadgk` method of the
`QuadGK.jl` package and the keyword arguments `kws` are passed
directly onto `quadgk`.

Note: If the function `g` contains additional integrable singularities,
the user should manually split the integration interval around them,
since currently there is no way of passing integration break points
onto `quadgk`.

## Arguments
- `g`: The function to integrate.
- `a`: The lower bound of the interval.
- `b`: The upper bound of the interval.
- `x0`: The location of the singularity. Default value is the origin.
- `kws...`: Additional keyword arguments accepted by `quadgk`.

## Returns
A tuple `(I, E)` containing the approximated integral `I` and
an estimated upper bound on the absolute error `E`.

## Throws
- `ArgumentError`: If the interval `[a, b]` does not include the singularity `x0`.

## Examples
```julia
julia> cauchy_quadgk(x -> 1/(x+2), -1.0, 1.0)
(-0.549306144334055, 9.969608472104596e-12)

julia> cauchy_quadgk(x -> x^2, 0.0, 2.0, 1.0)
(4.0, 2.220446049250313e-16)
```
"""
function cauchy_quadgk(g, a, b, x0=zero(promote_type(typeof(a),typeof(b))); kws...)
    a < x0 < b || throw(ArgumentError("domain must include the singularity"))
    g₀ = g(x0)
    g₀int = (b-x0) == -(a-x0) ? zero(g₀) : g₀ * log(abs((b-x0)/(a-x0))) / (b - a)
    return quadgk(x -> (g(x)-g₀)/(x-x0) + g₀int, a, x0, b; kws...)
end

"""
    hadamard_quadgk(g, g′, a, b, x0=0; kws...)

Computes the Hadamard finite part of the integral of a function `g`
with a singularity at `x0` over the interval `[a, b]`, that is
```math
\\mathcal{H}\\int_a^b\\frac{g(x)}{(x-x_0)^2}\\mathrm{d}x
```
Note that `x0` must be contained in the interval `[a, b]`.
The actual integration is performed by the `quadgk` method of the
`QuadGK.jl` package and the keyword arguments `kws` are passed
directly onto `quadgk`.

Note: If the function `g′` contains additional integrable singularities,
the user should manually split the integration interval around them,
since currently there is no way of passing integration break points
onto `quadgk`.

## Arguments
- `g`: The function to integrate.
- `g′`: The derivative of the function `g`.
- `a`: The lower bound of the interval.
- `b`: The upper bound of the interval.
- `x0`: The location of the singularity. Defaults value is the origin.
- `kws...`: Additional keyword arguments accepted by `quadgk`.

## Returns
A tuple `(I, E...)` containing the approximated Hadamard finite part integral `I` and
an additional error estimate `E` from the quadrature method.

## Throws
- `ArgumentError`: If the interval `[a, b]` does not include the singularity `x0`.

## Examples
```julia
julia> hadamard_quadgk(x -> log(x+1), x -> 1/(x+1), 0.0, 2.0, 1.0)
(-1.6479184330021648, 9.969608472104596e-12)
```
"""
function hadamard_quadgk(g, g′, a, b, x0=zero(promote_type(typeof(a),typeof(b))); kws...)
    a < x0 < b || throw(ArgumentError("domain must include the singularity"))
    I0 = g(a)/(a-x0) - g(b)/(b-x0)
    I1 = cauchy_quadgk(g′, a, b, x0; kws...)
    return (I0 + I1[1], I1[2:end]...)
end