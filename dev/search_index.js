var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = SpectralDensities","category":"page"},{"location":"#SpectralDensities","page":"Home","title":"SpectralDensities","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for SpectralDensities.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [SpectralDensities]","category":"page"},{"location":"#SpectralDensities.AbstractSD","page":"Home","title":"SpectralDensities.AbstractSD","text":"abstract type AbstractSD\n\nAbstractSD represents an abstract bath spectral density.\n\nAny subtype of AbstractSD must at least define either sd or sdoverω for the new type.\n\n\n\n\n\n","category":"type"},{"location":"#SpectralDensities.LorentzianSD","page":"Home","title":"SpectralDensities.LorentzianSD","text":"struct LorentzianSD <: AbstractSD\n\nLorentzianSD represents a Lorentzian spectral density. It is characterized by an amplitude α representing the strength of the coupling, the peak centre frequency ω0, and the width of the Lorentzian peak Γ.  That is\n\nJ(omega) = fracalphaGammapifracomega(omega^2 - omega_0^2)^2 + omega^2Gamma^2\n\nFields\n\nα::Float64: The amplitude α, indicating the strength of the coupling.\nω0::Float64: The centre frequency of the Lorentzian peak.\nΓ::Float64: The width of the Lorentzian peak.\n\n\n\n\n\n","category":"type"},{"location":"#SpectralDensities.LorentzianSD-Tuple{Any, Any, Any}","page":"Home","title":"SpectralDensities.LorentzianSD","text":"LorentzianSD(α, ω0, Γ)\n\nConstruct a Lorentzian spectral density with the given amplitude α, centre frequency ω0, and width Γ.\n\nArguments\n\nα: The amplitude α, indicating the strength of the coupling.\nω0: The centre frequency of the Lorentzian peak.\nΓ: The width of the Lorentzian peak.\n\nReturns\n\nAn instance of the LorentzianSD struct representing the Lorentzian spectral density.\n\n\n\n\n\n","category":"method"},{"location":"#SpectralDensities.OhmicSD","page":"Home","title":"SpectralDensities.OhmicSD","text":"struct OhmicSD <: AbstractSD\n\nOhmicSD represents an Ohmic spectral density. It is characterized by an amplitude α representing the strength of the Ohmic coupling. That is\n\nJ(omega) = alphaomega\n\nFields\n\nα::Float64: The amplitude α, indicating the strength of the Ohmic coupling.\n\n\n\n\n\n","category":"type"},{"location":"#SpectralDensities.OhmicSD-Tuple{Any}","page":"Home","title":"SpectralDensities.OhmicSD","text":"OhmicSD(α)\n\nConstruct an Ohmic spectral density with amplitude α.\n\nArguments\n\nα: The amplitude α, indicating the strength of the Ohmic coupling.\n\nReturns\n\nAn instance of the OhmicSD struct representing the Ohmic spectral density.\n\n\n\n\n\n","category":"method"},{"location":"#SpectralDensities.PolySD","page":"Home","title":"SpectralDensities.PolySD","text":"struct PolySD <: AbstractSD\n\nPolySD represents a polynomial spectral density. It is characterized by an amplitude α representing the strength of the coupling and the polynomial degree n. That is\n\nJ(omega) = alphaomega^n\n\nFields\n\nα::Float64: The amplitude α, indicating the strength of the coupling.\nn::Int: The polynomial degree.\n\n\n\n\n\n","category":"type"},{"location":"#SpectralDensities.PolySD-Tuple{Any, Int64}","page":"Home","title":"SpectralDensities.PolySD","text":"PolySD(α, n::Int)\n\nConstruct a polynomial spectral density with the given amplitude α and degree n.\n\nArguments\n\nα: The amplitude α, indicating the strength of the coupling.\nn::Int: The polynomial degree.\n\nReturns\n\nAn instance of the PolySD struct representing the polynomial spectral density.\n\n\n\n\n\n","category":"method"},{"location":"#SpectralDensities.UnderdampedSD","page":"Home","title":"SpectralDensities.UnderdampedSD","text":"struct UnderdampedSD <: AbstractSD\n\nRepresents an underdamped spectral density. This is just an alias for LorentzianSD.\n\n\n\n\n\n","category":"type"},{"location":"#SpectralDensities.correlations-Tuple{AbstractSD, Any, Any}","page":"Home","title":"SpectralDensities.correlations","text":"correlations(J::AbstractSD, τ, β)\n\nCalculate the correlation function for a spectral density J at a given time delay τ and inverse temperature β.\n\nArguments\n\nJ::AbstractSD: The spectral density.\nτ: The time delay at which the correlation function is calculated.\nβ: The inverse temperature.\n\nReturns\n\nThe correlation function for the spectral density J at the given time delay τ and inverse temperature β.\n\n\n\n\n\n","category":"method"},{"location":"#SpectralDensities.reorganisation_energy-Tuple{AbstractSD}","page":"Home","title":"SpectralDensities.reorganisation_energy","text":"reorganisation_energy(J::AbstractSD)\n\nCalculate the reorganization energy of a given spectral density J, i.e.\n\nint_0^infty fracJ(omega)omega mathrmdomega \n\nArguments\n\nJ::AbstractSD: The spectral density.\n\nReturns\n\nThe reorganization energy of the spectral density J.\n\n\n\n\n\n","category":"method"},{"location":"#SpectralDensities.sd-Union{Tuple{T}, Tuple{T, Any}} where T<:AbstractSD","page":"Home","title":"SpectralDensities.sd","text":"sd(J::T, ω) where T <: AbstractSD\n\nEvaluate the spectral density represented by J at a given frequency ω, i.e. J(ω).\n\nArguments\n\nJ::T: The spectral density.\nω: The frequency at which the spectral density is evaluated.\n\nReturns\n\nThe spectral density J at the frequency ω.\n\n\n\n\n\n","category":"method"},{"location":"#SpectralDensities.sdoverω-Union{Tuple{T}, Tuple{T, Any}} where T<:AbstractSD","page":"Home","title":"SpectralDensities.sdoverω","text":"sdoverω(J::T, ω) where T <: AbstractSD\n\nEvaluate the spectral density represented by J divided by a given frequency ω, i.e. J(ω)/ω.\n\nArguments\n\nJ::T: The spectral density.\nω: The frequency at which the spectral density is evaluated.\n\nReturns\n\nThe spectral density J(ω) divided by ω.\n\n\n\n\n\n","category":"method"}]
}
