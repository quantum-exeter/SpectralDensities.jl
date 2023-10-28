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

"""
    kramers_kronig(f, ω; cutoff=Inf, type=:real)

Use the Kramers-Kronig relations to compute the analytic continuation of the function `f` at point `ω`.
That is, if `type == :real` compute
```math
\\frac{1}{\\pi}\\int_{-\\infty}^{\\infty} \\frac{f(\\omega')}{\\omega'-\\omega} \\mathrm{d}\\omega',
```
and if `type == :imag` compute
```math
-\\frac{1}{\\pi}\\int_{-\\infty}^{\\infty} \\frac{f(\\omega')}{\\omega'-\\omega} \\mathrm{d}\\omega'.
```

# Arguments
- `f`: A function that is either the real or imaginary part of an analytic complex function.
- `ω`: The point where to evaluate the analytic continuation of `f`.
- `cutoff`: (optional, default=Inf) A cutoff that bounds the range of integration.
- `type::Symbol`: (optional, default=:real) If `:real`, take `f` as the imaginary part and calculate the real part of the analytic continuation.
If `:imag`, take `f` as the real part and calculate the imaginary part of the analytic continuation.

# Returns
- The real or imaginary part of the analytic continuation of `f` evaluated at `ω`.

"""
function kramers_kronig(f, ω; cutoff=Inf, type=:real)
    sign = type == :real ? 1 : -1
    return cauchy_quadgk(u -> sign*f(u + ω)/π, -cutoff, +cutoff)[1]
end