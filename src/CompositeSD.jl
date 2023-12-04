"""
    struct CompositeSD <: AbstractSD

CompositeSD represents a composite spectral density consisting
of the sum of multiple individual spectral densities.

# Fields
- `Jlist::Vector{AbstractSD}`: A vector containing individual spectral densties.

"""
struct CompositeSD <: AbstractSD
    Jlist::Vector{AbstractSD}
end

sd(J::CompositeSD, ω) = sum([sd(Ji,ω) for Ji in J.Jlist])

sdoverω(J::CompositeSD, ω) = sum([sdoverω(Ji,ω) for Ji in J.Jlist])

reorganisation_energy(J::CompositeSD) = sum([reorganisation_energy(Ji) for Ji in J.Jlist])

memory_kernel(J::CompositeSD, τ; ωcutoff=Inf) = sum([memory_kernel(Ji,τ;ωcutoff) for Ji in J.Jlist])

memory_kernel_ft(J::CompositeSD, ω) = sum([memory_kernel_ft(Ji,ω) for Ji in J.Jlist])

real_memory_kernel_ft(J::CompositeSD, ω) = sum([real_memory_kernel_ft(Ji,ω) for Ji in J.Jlist])

imag_memory_kernel_ft(J::CompositeSD, ω) = sum([imag_memory_kernel_ft(Ji,ω) for Ji in J.Jlist])

frequency_cutoff(J::CompositeSD; tol=eps()) = maxmium([frequency_cutoff(Ji;tol) for Ji in J.Jlist])

frequency_step(J::CompositeSD; tol=eps()) = minimum([frequency_step(Ji;tol) for Ji in J.Jlist])

function Base.:+(l::CompositeSD, r::CompositeSD)
    return CompositeSD([l.Jlist; r.Jlist])
end

function Base.:+(l::CompositeSD, r::AbstractSD)
    return CompositeSD([l.Jlist; r])
end

function Base.:+(l::AbstractSD, r::CompositeSD)
    return CompositeSD([l; r.Jlist])
end

function Base.:+(l::AbstractSD, r::AbstractSD)
    return CompositeSD([l, r])
end