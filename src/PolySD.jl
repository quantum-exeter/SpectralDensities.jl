"""
    struct PolySD <: AbstractSD

PolySD represents a polynomial spectral density.
It is characterized by an amplitude `α` representing the strength of the coupling
and the polynomial degree `n`. That is
```math
J(\\omega) = \\alpha\\omega^n
```

# Fields
- `α::Float64`: The amplitude `α`, indicating the strength of the coupling.
- `n::Int`: The polynomial degree.

"""
struct PolySD <: AbstractSD
    α::Float64
    n::Int
end

"""
    PolySD(α, n::Int)

Construct a polynomial spectral density with the given amplitude `α` and degree `n`.

# Arguments
- `α`: The amplitude `α`, indicating the strength of the coupling.
- `n::Int`: The polynomial degree.

# Returns
- An instance of the `PolySD` struct representing the polynomial spectral density.

"""
PolySD(α, n::Int) = PolySD(float(α), n)
 
sdoverω(J::PolySD, ω) = J.α*(ω^(J.n-1))

reorganisation_energy(::PolySD) = Inf