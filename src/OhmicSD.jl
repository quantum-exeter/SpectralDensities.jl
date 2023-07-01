"""
    struct OhmicSD <: AbstractSD

OhmicSD represents an Ohmic spectral density.
It is characterized by an amplitude `α` representing the strength of the Ohmic coupling.
That is
```math
J(\\omega) = \\alpha\\omega
```

# Fields
- `α::Float64`: The amplitude `α`, indicating the strength of the Ohmic coupling.

"""
struct OhmicSD <: AbstractSD 
    α::Float64
end
 
"""
    OhmicSD(α)

Construct an Ohmic spectral density with amplitude `α`.

# Arguments
- `α`: The amplitude `α`, indicating the strength of the Ohmic coupling.

# Returns
- An instance of the `OhmicSD` struct representing the Ohmic spectral density.

"""
OhmicSD(α) = OhmicSD(float(α))
 
sdoverω(J::OhmicSD, ω) = J.α

reorganisation_energy(::OhmicSD) = Inf