var documenterSearchIndex = {"docs":
[{"location":"reference/#API-Reference","page":"API Reference","title":"API Reference","text":"","category":"section"},{"location":"reference/#SpectralDensities","page":"API Reference","title":"SpectralDensities","text":"","category":"section"},{"location":"reference/","page":"API Reference","title":"API Reference","text":"Modules = [SpectralDensities]","category":"page"},{"location":"reference/#SpectralDensities.AbstractSD","page":"API Reference","title":"SpectralDensities.AbstractSD","text":"abstract type AbstractSD\n\nAbstractSD represents an abstract bath spectral density.\n\nAny subtype of AbstractSD must at least define either sd or sdoverω for the new type.\n\n\n\n\n\n","category":"type"},{"location":"reference/#SpectralDensities.CompositeSD","page":"API Reference","title":"SpectralDensities.CompositeSD","text":"struct CompositeSD <: AbstractSD\n\nCompositeSD represents a composite spectral density consisting of the sum of multiple individual spectral densities.\n\nFields\n\nJlist::Vector{AbstractSD}: A vector containing individual spectral densties.\n\n\n\n\n\n","category":"type"},{"location":"reference/#SpectralDensities.DebyeSD","page":"API Reference","title":"SpectralDensities.DebyeSD","text":"struct DebyeSD <: AbstractSD\n\nDebyeSD represents a Debye spectral density. It is characterized by an amplitude α representing the strength of the coupling and the cutoff frequency ωc. That is\n\nJ(omega) = frac2alphapifracomegaomega_c^2omega^2 + omega_c^2\n\nFields\n\nα::Float64: The amplitude α, indicating the strength of the coupling.\nωc::Float64: The cutoff frequency.\n\n\n\n\n\n","category":"type"},{"location":"reference/#SpectralDensities.DebyeSD-Tuple{Real, Real}","page":"API Reference","title":"SpectralDensities.DebyeSD","text":"DebyeSD(α::Real, ωc::Real)\n\nConstruct a Debye spectral density with the given amplitude α and cutoff frequency ωc.\n\nArguments\n\nα::Real: The amplitude α, indicating the strength of the coupling.\nωc::Real: The cutoff frequency.\n\nReturns\n\nAn instance of the DebyeSD struct representing the Debye spectral density.\n\n\n\n\n\n","category":"method"},{"location":"reference/#SpectralDensities.ExponentialCutoffSD","page":"API Reference","title":"SpectralDensities.ExponentialCutoffSD","text":"struct ExponentialCutoffSD{T <: AbstractSD} <: AbstractSD\n\nExponentialCutoffSD represents a spectral density with an exponential frequency cutoff. It is parameterized by the underlying spectral density J, and the frequency cutoff ωcutoff. That is\n\nJ_mathrmexp(omega) = J(omega)e^-omegaomega_c\n\nFields\n\nJ::T: The underlying spectral density.\nωcutoff::Float64: The frequency cutoff.\n\n\n\n\n\n","category":"type"},{"location":"reference/#SpectralDensities.ExponentialCutoffSD-Tuple{Any, Any}","page":"API Reference","title":"SpectralDensities.ExponentialCutoffSD","text":"ExponentialCutoffSD(J::AbstractSD, ωcutoff)\n\nConstruct a spectral density with an exponential frequency cutoff using the underlying spectral density J and the given frequency cutoff ωcutoff.\n\nArguments\n\nJ::AbstractSD: The underlying spectral density.\nωcutoff: The frequency cutoff.\n\nReturns\n\nAn instance of the ExponentialCutoffSD struct representing the spectral density with an exponential frequency cutoff.\n\n\n\n\n\n","category":"method"},{"location":"reference/#SpectralDensities.GaussianCutoffSD","page":"API Reference","title":"SpectralDensities.GaussianCutoffSD","text":"struct GaussianCutoffSD{T <: AbstractSD} <: AbstractSD\n\nGaussianCutoffSD represents a spectral density with a Gaussian frequency cutoff. It is parameterized by the underlying spectral density J, and the frequency cutoff ωcutoff. That is\n\nJ_mathrmgauss(omega) = J(omega)e^-omega^2omega_c^2\n\nFields\n\nJ::T: The underlying spectral density.\nωcutoff::Float64: The frequency cutoff.\n\n\n\n\n\n","category":"type"},{"location":"reference/#SpectralDensities.GaussianCutoffSD-Tuple{Any, Any}","page":"API Reference","title":"SpectralDensities.GaussianCutoffSD","text":"GaussianCutoffSD(J::AbstractSD, ωcutoff)\n\nConstruct a spectral density with a Gaussian frequency cutoff using the underlying spectral density J and the given frequency cutoff ωcutoff.\n\nArguments\n\nJ::AbstractSD: The underlying spectral density.\nωcutoff: The frequency cutoff.\n\nReturns\n\nAn instance of the GaussianCutoffSD struct representing the spectral density with a Gaussian frequency cutoff.\n\n\n\n\n\n","category":"method"},{"location":"reference/#SpectralDensities.HardCutoffSD","page":"API Reference","title":"SpectralDensities.HardCutoffSD","text":"struct HardCutoffSD{T <: AbstractSD} <: AbstractSD\n\nHardCutoffSD represents a spectral density with a hard frequency cutoff. It is parameterized by the underlying spectral density J, and the frequency cutoff ωcutoff. That is\n\nJ_mathrmhard(omega) = J(omega)Theta(omega_c - omega)\n\nwhere Theta is the Heaviside theta function.\n\nFields\n\nJ::T: The underlying spectral density.\nωcutoff::Float64: The frequency cutoff.\n\n\n\n\n\n","category":"type"},{"location":"reference/#SpectralDensities.HardCutoffSD-Tuple{AbstractSD, Any}","page":"API Reference","title":"SpectralDensities.HardCutoffSD","text":"HardCutoffSD(J::AbstractSD, ωcutoff)\n\nConstruct a spectral density with a hard frequency cutoff using the underlying spectral density J and the given frequency cutoff ωcutoff.\n\nArguments\n\nJ::AbstractSD: The underlying spectral density.\nωcutoff::Float64: The frequency cutoff.\n\nReturns\n\nAn instance of the HardCutoffSD struct representing the spectral density with a hard frequency cutoff.\n\n\n\n\n\n","category":"method"},{"location":"reference/#SpectralDensities.InversePolyKernelSD","page":"API Reference","title":"SpectralDensities.InversePolyKernelSD","text":"struct InversePolyKernelSD <: AbstractSD\n\nInversePolyKernelSD represents a spectral density whose corresponding memory kernel in frequency space is given by one over a polynomial in frequency. It is characterized by the degree of the polynomial deg and the list of coefficients coefs (in ascending order of powers). That is,\n\nJ(omega) = frac1pimathrmImleftfrac1sum_k=0^mathrmdegI_kmathrmcoeffsk+1omega^kright\n\nwhere math I_k = 1 if k is even, and math I_k = i if k is odd.\n\nFields\n\ndeg::Int64: The degree of the polynomial.\ncoeffs::AbstractVector{ComplexF64}: Coefficients of the polynomial (in ascending order of powers). The length of the vector should be deg + 1.\n\n\n\n\n\n","category":"type"},{"location":"reference/#SpectralDensities.InversePolyKernelSD-Tuple{DebyeSD}","page":"API Reference","title":"SpectralDensities.InversePolyKernelSD","text":"InversePolyKernelSD(J::DebyeSD)\n\nConstruct a InversePolyKernelSD spectral density from a Debye spectral density.\n\nArguments\n\nJ::DebyeSD: The Debye spectral density.\n\nReturns\n\nAn instance of the InversePolyKernelSD that represents the Debye spectral density.\n\n\n\n\n\n","category":"method"},{"location":"reference/#SpectralDensities.InversePolyKernelSD-Tuple{LorentzianSD}","page":"API Reference","title":"SpectralDensities.InversePolyKernelSD","text":"InversePolyKernelSD(J::LorentzianSD)\n\nConstruct a InversePolyKernelSD spectral density from a Lorentzian spectral density.\n\nArguments\n\nJ::LorentzianSD: The Lorentzian spectral density.\n\nReturns\n\nAn instance of the InversePolyKernelSD that represents the Lorentzian spectral density.\n\n\n\n\n\n","category":"method"},{"location":"reference/#SpectralDensities.InversePolyKernelSD-Tuple{Vector{<:Real}}","page":"API Reference","title":"SpectralDensities.InversePolyKernelSD","text":"InversePolyKernelSD(coeffs::AbstractVector)\n\nConstruct a InversePolyKernelSD spectral density with the given coefficients.\n\nArguments\n\ncoeffs::AbstractVector: A vector containing the coefficients of the polynomial (in ascending order of powers).\n\nReturns\n\nAn instance of the InversePolyKernelSD struct representing the spectral density.\n\n\n\n\n\n","category":"method"},{"location":"reference/#SpectralDensities.LorentzDrudeSD","page":"API Reference","title":"SpectralDensities.LorentzDrudeSD","text":"struct LorentzDrudeSD <: AbstractSD\n\nRepresents a Lorentz-Drude spectral density. This is just an alias for DebyeSD.\n\n\n\n\n\n","category":"type"},{"location":"reference/#SpectralDensities.LorentzianSD","page":"API Reference","title":"SpectralDensities.LorentzianSD","text":"struct LorentzianSD <: AbstractSD\n\nLorentzianSD represents a Lorentzian spectral density. It is characterized by an amplitude α representing the strength of the coupling, the peak centre frequency ω0, and the width of the Lorentzian peak Γ.  That is\n\nJ(omega) = fracalphaGammapifracomega(omega^2 - omega_0^2)^2 + omega^2Gamma^2\n\nFields\n\nα::Float64: The amplitude α, indicating the strength of the coupling.\nω0::Float64: The centre frequency of the Lorentzian peak.\nΓ::Float64: The width of the Lorentzian peak.\n\n\n\n\n\n","category":"type"},{"location":"reference/#SpectralDensities.LorentzianSD-Tuple{Real, Real, Real}","page":"API Reference","title":"SpectralDensities.LorentzianSD","text":"LorentzianSD(α::Real, ω0::Real, Γ::Real)\n\nConstruct a Lorentzian spectral density with the given amplitude α, centre frequency ω0, and width Γ.\n\nArguments\n\nα::Real: The amplitude α, indicating the strength of the coupling.\nω0::Real: The centre frequency of the Lorentzian peak.\nΓ::Real: The width of the Lorentzian peak.\n\nReturns\n\nAn instance of the LorentzianSD struct representing the Lorentzian spectral density.\n\n\n\n\n\n","category":"method"},{"location":"reference/#SpectralDensities.OhmicSD","page":"API Reference","title":"SpectralDensities.OhmicSD","text":"struct OhmicSD <: AbstractSD\n\nOhmicSD represents an Ohmic spectral density. It is characterized by an amplitude α representing the strength of the Ohmic coupling. That is\n\nJ(omega) = alphaomega\n\nFields\n\nα::Float64: The amplitude α, indicating the strength of the Ohmic coupling.\n\n\n\n\n\n","category":"type"},{"location":"reference/#SpectralDensities.OhmicSD-Tuple{Real}","page":"API Reference","title":"SpectralDensities.OhmicSD","text":"OhmicSD(α::Real)\n\nConstruct an Ohmic spectral density with amplitude α.\n\nArguments\n\nα::Real: The amplitude α, indicating the strength of the Ohmic coupling.\n\nReturns\n\nAn instance of the OhmicSD struct representing the Ohmic spectral density.\n\n\n\n\n\n","category":"method"},{"location":"reference/#SpectralDensities.OverdampedSD","page":"API Reference","title":"SpectralDensities.OverdampedSD","text":"struct OverdampedSD <: AbstractSD\n\nRepresents an overdamped spectral density. This is just an alias for DebyeSD.\n\n\n\n\n\n","category":"type"},{"location":"reference/#SpectralDensities.PolySD","page":"API Reference","title":"SpectralDensities.PolySD","text":"struct PolySD <: AbstractSD\n\nPolySD represents a polynomial spectral density. It is characterized by an amplitude α representing the strength of the coupling and the polynomial degree n. That is\n\nJ(omega) = alphaomega^n\n\nFields\n\nα::Float64: The amplitude α, indicating the strength of the coupling.\nn::Int: The polynomial degree.\n\n\n\n\n\n","category":"type"},{"location":"reference/#SpectralDensities.PolySD-Tuple{Real, Int64}","page":"API Reference","title":"SpectralDensities.PolySD","text":"PolySD(α::Real, n::Int)\n\nConstruct a polynomial spectral density with the given amplitude α and degree n.\n\nArguments\n\nα::Real: The amplitude α, indicating the strength of the coupling.\nn::Int: The polynomial degree.\n\nReturns\n\nAn instance of the PolySD struct representing the polynomial spectral density.\n\n\n\n\n\n","category":"method"},{"location":"reference/#SpectralDensities.UnderdampedSD","page":"API Reference","title":"SpectralDensities.UnderdampedSD","text":"struct UnderdampedSD <: AbstractSD\n\nRepresents an underdamped spectral density. This is just an alias for LorentzianSD.\n\n\n\n\n\n","category":"type"},{"location":"reference/#SpectralDensities.correlations-Tuple{AbstractSD, Any, Any}","page":"API Reference","title":"SpectralDensities.correlations","text":"correlations(J::AbstractSD, τ, β)\n\nCalculate the correlation function for a spectral density J at a given time delay τ and inverse temperature β, that is\n\nmathcalC(tau) =\nint_0^infty J(omega)cothleft(frachbaromegabeta2right)cos(omegatau)mathrmdomega\n-iint_0^infty J(omega)sin(omegatau)mathrmdomega\n\nArguments\n\nJ::AbstractSD: The spectral density.\nτ: The time delay at which the correlation function is calculated.\nβ: The inverse temperature.\n\nReturns\n\nThe correlation function for the spectral density J at the given time delay τ and inverse temperature β.\n\n\n\n\n\n","category":"method"},{"location":"reference/#SpectralDensities.correlations_imag-Tuple{AbstractSD, Any}","page":"API Reference","title":"SpectralDensities.correlations_imag","text":"correlations_imag(J::AbstractSD, τ, β)\n\nCalculate the imaginary part of the correlation function for a spectral density J at a given time delay τ, that is\n\nmathrmRemathcalC(tau) = -int_0^infty J(omega)sin(omegatau)mathrmdomega\n\nArguments\n\nJ::AbstractSD: The spectral density.\nτ: The time delay at which the correlation function is calculated.\n\nReturns\n\nThe imaginary part of the correlation function for the spectral density J at\n\nthe given time delay τ.\n\n\n\n\n\n","category":"method"},{"location":"reference/#SpectralDensities.correlations_real-Tuple{AbstractSD, Any, Any}","page":"API Reference","title":"SpectralDensities.correlations_real","text":"correlations_real(J::AbstractSD, τ, β)\n\nCalculate the real part of the correlation function for a spectral density J at a given time delay τ and inverse temperature β, that is\n\nmathrmRemathcalC(tau) = int_0^infty J(omega)cothleft(frachbaromegabeta2right)cos(omegatau)mathrmdomega\n\nArguments\n\nJ::AbstractSD: The spectral density.\nτ: The time delay at which the correlation function is calculated.\nβ: The inverse temperature.\n\nReturns\n\nThe real part of the correlation function for the spectral density J at the\n\ngiven time delay τ and inverse temperature β.\n\n\n\n\n\n","category":"method"},{"location":"reference/#SpectralDensities.frequency_cutoff-Tuple{AbstractSD}","page":"API Reference","title":"SpectralDensities.frequency_cutoff","text":"frequency_cutoff(J::AbstractSD; tol=eps())\n\nReturn the frequency cutoff of the spectral density J to accuracy set by the tolerance parameter tol.\n\nArguments\n\nJ::AbstractSD: The spectral density.\ntol: The tolerance at which to truncate the spectral density.\n\nReturns\n\nThe frequency cutoff for the desired tolerance.\n\n\n\n\n\n","category":"method"},{"location":"reference/#SpectralDensities.frequency_step-Tuple{AbstractSD}","page":"API Reference","title":"SpectralDensities.frequency_step","text":"frequency_step(J::AbstractSD; tol=eps())\n\nCalculate an appropriate frequency step for accurate numerical discretisation of the spectral density J.\n\nArguments\n\nJ::AbstractSD: The spectral density.\ntol::Real: (optional, default=eps()) The tolerance used to determine the appropriate frequency setp.\n\nReturns\n\nThe calculated frequency step for discretisation at the given tolerance.\n\n\n\n\n\n","category":"method"},{"location":"reference/#SpectralDensities.imag_memory_kernel_ft-Tuple{AbstractSD, Any}","page":"API Reference","title":"SpectralDensities.imag_memory_kernel_ft","text":"imag_memory_kernel_ft(J::AbstractSD, ω)\n\nCalculate the imaginary part of the Fourier-transform of the  memory kernel for a spectral density J at a given frequency ω.\n\nArguments\n\nJ::AbstractSD: The spectral density.\nω: The frequency at which the imaginary part of the Fourier-transform of the memory kernel is evaluated.\n\nReturns\n\nThe imaginary part of the Fourier-transform of the memory kernel for the spectral density J at the given frequency ω.\n\n\n\n\n\n","category":"method"},{"location":"reference/#SpectralDensities.memory_kernel-Tuple{AbstractSD, Any}","page":"API Reference","title":"SpectralDensities.memory_kernel","text":"memory_kernel(J::AbstractSD, τ)\n\nCalculate the memory kernel for a spectral density J at a given time delay τ, that is\n\nmathcalK(tau) = 2Theta(tau)int_0^infty J(omega)sin(omegatau)mathrmdomega\n\nwhere Theta is the Heavisde theta function.\n\nArguments\n\nJ::AbstractSD: The spectral density.\nτ: The time delay at which the memory kernel is evaluated.\nωcutoff: (optinal, default: Inf) Frequency cutoff to be used when calculating the correlation function.\n\nReturns\n\nThe memory kernel for the spectral density J at the given time delay τ.\n\n\n\n\n\n","category":"method"},{"location":"reference/#SpectralDensities.memory_kernel_ft-Tuple{AbstractSD, Any}","page":"API Reference","title":"SpectralDensities.memory_kernel_ft","text":"memory_kernel_ft(J::AbstractSD, ω)\n\nCalculate the Fourier-transform of the  memory kernel for a spectral density J at a given frequency ω.\n\nArguments\n\nJ::AbstractSD: The spectral density.\nω: The frequency at which the Fourier-transform of the memory kernel is evaluated.\n\nReturns\n\nThe Fourier-transform of the memory kernel for the spectral density J at the given frequency ω.\n\n\n\n\n\n","category":"method"},{"location":"reference/#SpectralDensities.real_memory_kernel_ft-Tuple{AbstractSD, Any}","page":"API Reference","title":"SpectralDensities.real_memory_kernel_ft","text":"real_memory_kernel_ft(J::AbstractSD, ω)\n\nCalculate the real part of the Fourier-transform of the  memory kernel for a spectral density J at a given frequency ω.\n\nArguments\n\nJ::AbstractSD: The spectral density.\nω: The frequency at which the real part of the Fourier-transform of the memory kernel is evaluated.\n\nReturns\n\nThe real part of the Fourier-transform of the memory kernel for the spectral density J at the given frequency ω.\n\n\n\n\n\n","category":"method"},{"location":"reference/#SpectralDensities.reorganisation_energy-Tuple{AbstractSD}","page":"API Reference","title":"SpectralDensities.reorganisation_energy","text":"reorganisation_energy(J::AbstractSD)\n\nCalculate the reorganization energy of a given spectral density J, i.e.\n\nint_0^infty fracJ(omega)omega mathrmdomega \n\nArguments\n\nJ::AbstractSD: The spectral density.\n\nReturns\n\nThe reorganization energy of the spectral density J.\n\n\n\n\n\n","category":"method"},{"location":"reference/#SpectralDensities.sd-Union{Tuple{T}, Tuple{T, Any}} where T<:AbstractSD","page":"API Reference","title":"SpectralDensities.sd","text":"sd(J::T, ω) where T <: AbstractSD\n\nEvaluate the spectral density represented by J at a given frequency ω, i.e. J(ω).\n\nArguments\n\nJ::T: The spectral density.\nω: The frequency at which the spectral density is evaluated.\n\nReturns\n\nThe spectral density J at the frequency ω.\n\n\n\n\n\n","category":"method"},{"location":"reference/#SpectralDensities.sdoverω-Union{Tuple{T}, Tuple{T, Any}} where T<:AbstractSD","page":"API Reference","title":"SpectralDensities.sdoverω","text":"sdoverω(J::T, ω) where T <: AbstractSD\n\nEvaluate the spectral density represented by J divided by a given frequency ω, i.e. J(ω)/ω.\n\nArguments\n\nJ::T: The spectral density.\nω: The frequency at which the spectral density is evaluated.\n\nReturns\n\nThe spectral density J(ω) divided by ω.\n\n\n\n\n\n","category":"method"},{"location":"reference/#WeakCoupling","page":"API Reference","title":"WeakCoupling","text":"","category":"section"},{"location":"reference/","page":"API Reference","title":"API Reference","text":"Modules = [WeakCoupling]","category":"page"},{"location":"reference/#SpectralDensities.WeakCoupling.weak_coupling_Δ-Tuple{AbstractSD, Any, Any}","page":"API Reference","title":"SpectralDensities.WeakCoupling.weak_coupling_Δ","text":"weak_coupling_Δ(J::AbstractSD, ωB, β; ωcutoff=Inf, ħ=one(ωB))\n\nCalculate the weak-coupling coefficient Δ for the spectral density J, system Bohr frequency ωB, and inverse temperature β, defined as\n\nDelta_beta(omega_mathrmB) = 2omega_mathrmBint_0^infty\n    J(omega)frac1omega^2-omega_mathrmB^2\n    cothleft(fracbetahbaromega2right) mathrmdomega\n\nSee: J.D. Cresser, J. Anders, Phys. Rev. Lett. 127, 250601 (2021).\n\nArguments\n\nJ::AbstractSD: The spectral density.\nωB: The system Bohr frequency of interest.\nβ: The inverse temperature.\nωcutoff: (optional, default=Inf) Frequency cutoff to help with the convergence of the integration.\nħ: The value of the reduced Planck constant. Default is 1.\n\nReturns\n\nThe weak-coupling coefficient Δ for the spectral density J, system Bohr frequency ωB, and inverse temperature β.\n\n\n\n\n\n","category":"method"},{"location":"reference/#SpectralDensities.WeakCoupling.weak_coupling_Δprime-Tuple{AbstractSD, Any, Any}","page":"API Reference","title":"SpectralDensities.WeakCoupling.weak_coupling_Δprime","text":"weak_coupling_Δprime(J::AbstractSD, ωB, β; ωcutoff=Inf, ħ=one(ωB))\n\nCalculate the weak-coupling coefficient Δ′ for the spectral density J, system Bohr frequency ωB, and inverse temperature β, defined as\n\nDelta_beta(omega_mathrmB) = 2int_0^infty\n    J(omega)frac(omega^2 + omega_mathrmB^2)(omega^2-omega_mathrmB^2)^2\n    cothleft(fracbetahbaromega2right) mathrmdomega\n\nSee: J.D. Cresser, J. Anders, Phys. Rev. Lett. 127, 250601 (2021).\n\nNote: The spectral density J must support automatic differentiation with ForwardDiff.jl.\n\nArguments\n\nJ::AbstractSD: The spectral density.\nωB: The system Bohr frequency of interest.\nβ: The inverse temperature.\nωcutoff: (optional, default=Inf) Frequency cutoff to help with the convergence of the integration.\nħ: The value of the reduced Planck constant. Default is 1.\n\nReturns\n\nThe weak-coupling coefficient Δ′ for the spectral density J, system Bohr frequency ωB, and inverse temperature β.\n\n\n\n\n\n","category":"method"},{"location":"reference/#SpectralDensities.WeakCoupling.weak_coupling_Σ-Tuple{AbstractSD, Any}","page":"API Reference","title":"SpectralDensities.WeakCoupling.weak_coupling_Σ","text":"weak_coupling_Σ(J::AbstractSD, ωB; ωcutoff=Inf)\n\nCalculate the weak-coupling coefficient Σ for the spectral density J and system Bohr frequency ωB, defined as\n\nSigma(omega_mathrmB) = 2int_0^infty\n    J(omega)fracomegaomega^2-omega_mathrmB^2 mathrmdomega\n\nSee: J.D. Cresser, J. Anders, Phys. Rev. Lett. 127, 250601 (2021).\n\nArguments\n\nJ::AbstractSD: The spectral denstiy.\nωB: The system Bohr frequency of interest.\nωcutoff: (optional, default=Inf) Frequency cutoff to help with the convergence of the integration.\n\nReturns\n\nThe weak-coupling coefficient Σ for the spectral density J and system Bohr frequency ωB.\n\n\n\n\n\n","category":"method"},{"location":"reference/#SpectralDensities.WeakCoupling.weak_coupling_Σprime-Tuple{AbstractSD, Any}","page":"API Reference","title":"SpectralDensities.WeakCoupling.weak_coupling_Σprime","text":"weak_coupling_Σprime(J::AbstractSD, ωB; ωcutoff=Inf)\n\nCalculate the weak-coupling coefficient Σ′ for the spectral density J and system Bohr frequency ωB, defined as\n\nSigma(omega_mathrmB) = 4omega_mathrmBint_0^infty\n    J(omega)fracomega(omega^2-omega_mathrmB^2)^2\n    mathrmdomega\n\nSee: J.D. Cresser, J. Anders, Phys. Rev. Lett. 127, 250601 (2021).\n\nNote: The spectral density J must support automatic differentiation with ForwardDiff.jl.\n\nArguments\n\nJ::AbstractSD: The spectral density.\nωB: The system Bohr frequency of interest.\nωcutoff: (optional, default=Inf) Frequency cutoff to help with the convergence of the integration.\n\nReturns\n\nThe weak-coupling coefficient Σ′ for the spectral density J and system Bohr frequency ωB.\n\n\n\n\n\n","category":"method"},{"location":"reference/#SingularIntegrals","page":"API Reference","title":"SingularIntegrals","text":"","category":"section"},{"location":"reference/","page":"API Reference","title":"API Reference","text":"Modules = [SingularIntegrals]","category":"page"},{"location":"reference/#SpectralDensities.SingularIntegrals.cauchy_quadgk","page":"API Reference","title":"SpectralDensities.SingularIntegrals.cauchy_quadgk","text":"cauchy_quadgk(g, a, b, x0=0; kws...)\n\nComputes the Cauchy principal value of the integral of a function g with singularity at x0 over the interval [a, b], that is\n\nmathrmPVint_a^bfracg(x)x-x_0mathrmdx\n\nNote that x0 must be contained in the interval [a, b]. The actual integration is performed by the quadgk method of the QuadGK.jl package and the keyword arguments kws are passed directly onto quadgk.\n\nNote: If the function g contains additional integrable singularities, the user should manually split the integration interval around them, since currently there is no way of passing integration break points onto quadgk.\n\nArguments\n\ng: The function to integrate.\na: The lower bound of the interval.\nb: The upper bound of the interval.\nx0: The location of the singularity. Default value is the origin.\nkws...: Additional keyword arguments accepted by quadgk.\n\nReturns\n\nA tuple (I, E) containing the approximated integral I and an estimated upper bound on the absolute error E.\n\nThrows\n\nArgumentError: If the interval [a, b] does not include the singularity x0.\n\nExamples\n\njulia> cauchy_quadgk(x -> 1/(x+2), -1.0, 1.0)\n(-0.549306144334055, 9.969608472104596e-12)\n\njulia> cauchy_quadgk(x -> x^2, 0.0, 2.0, 1.0)\n(4.0, 2.220446049250313e-16)\n\n\n\n\n\n","category":"function"},{"location":"reference/#SpectralDensities.SingularIntegrals.hadamard_quadgk","page":"API Reference","title":"SpectralDensities.SingularIntegrals.hadamard_quadgk","text":"hadamard_quadgk(g, g′, a, b, x0=0; kws...)\n\nComputes the Hadamard finite part of the integral of a function g with a singularity at x0 over the interval [a, b], that is\n\nmathcalHint_a^bfracg(x)(x-x_0)^2mathrmdx\n\nNote that x0 must be contained in the interval [a, b]. The actual integration is performed by the quadgk method of the QuadGK.jl package and the keyword arguments kws are passed directly onto quadgk.\n\nNote: If the function g′ contains additional integrable singularities, the user should manually split the integration interval around them, since currently there is no way of passing integration break points onto quadgk.\n\nArguments\n\ng: The function to integrate.\ng′: The derivative of the function g.\na: The lower bound of the interval.\nb: The upper bound of the interval.\nx0: The location of the singularity. Defaults value is the origin.\nkws...: Additional keyword arguments accepted by quadgk.\n\nReturns\n\nA tuple (I, E...) containing the approximated Hadamard finite part integral I and an additional error estimate E from the quadrature method.\n\nThrows\n\nArgumentError: If the interval [a, b] does not include the singularity x0.\n\nExamples\n\njulia> hadamard_quadgk(x -> log(x+1), x -> 1/(x+1), 0.0, 2.0, 1.0)\n(-1.6479184330021648, 9.969608472104596e-12)\n\n\n\n\n\n","category":"function"},{"location":"reference/#SpectralDensities.SingularIntegrals.kramers_kronig-Tuple{Any, Any}","page":"API Reference","title":"SpectralDensities.SingularIntegrals.kramers_kronig","text":"kramers_kronig(f, ω; cutoff=Inf, type=:real)\n\nUse the Kramers-Kronig relations to compute the analytic continuation of the function f at point ω. That is, if type == :real compute\n\nfrac1piint_-infty^infty fracf(omega)omega-omega mathrmdomega\n\nand if type == :imag compute\n\n-frac1piint_-infty^infty fracf(omega)omega-omega mathrmdomega\n\nArguments\n\nf: A function that is either the real or imaginary part of an analytic complex function.\nω: The point where to evaluate the analytic continuation of f.\ncutoff: (optional, default=Inf) A cutoff that bounds the range of integration.\ntype::Symbol: (optional, default=:real) If :real, take f as the imaginary part and calculate the real part of the analytic continuation.\n\nIf :imag, take f as the real part and calculate the imaginary part of the analytic continuation.\n\nReturns\n\nThe real or imaginary part of the analytic continuation of f evaluated at ω.\n\n\n\n\n\n","category":"method"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = SpectralDensities","category":"page"},{"location":"#SpectralDensities.jl","page":"Home","title":"SpectralDensities.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This package implements commonly used spectral densities for Open Quantum Systems and typical operations on them.","category":"page"},{"location":"#Features","page":"Home","title":"Features","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Features of SpectralDensities.jl include:","category":"page"},{"location":"","page":"Home","title":"Home","text":"Definition of most widely used spectral densities: Ohmic, Sub-Ohmic, Supra-Ohmic, Lorentzian (Underdamped), Debye (Overdamped)\nFlexibility to choose desidred cutoff functions: hard cutoffs, exponential cutoff, gaussian cutoff.\nCalculation of the correlation function of a given spectral density (with specialised methods for the pre-defined types)\nCalculation of reorganisation energy of a given spectral density (with specialised methods for the pre-defined types)\nCalculation of the memory kernel of a given spectral density, both in the time domain, and the imaginary part in the Fourier domain (with specialised methods for some of the pre-defined types)\nMethods to compute the spectral density integrals that appear in the weak coupling (2nd order expansion), both for the dynamics (see e.g. H. Breuer, F. Petruccione, \"The Theory of Open Quantum Systems\"), and the equilibrium mean-force state (see e.g. J.D. Cresser, J. Anders, Phys. Rev. Lett. 127, 250601 (2021)).","category":"page"},{"location":"#Quick-start","page":"Home","title":"Quick start","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The desired spectral density can be constructed by simply passing the desired parameters to their respective constructor. For example, for an Ohmic spectral density J(omega) = 3omega we have","category":"page"},{"location":"","page":"Home","title":"Home","text":"Johmic = OhmicSD(3)","category":"page"},{"location":"","page":"Home","title":"Home","text":"To add a ctuoff to a base spectral density, one simply passes it to the cutoff constructor. For example, to add an exponential cutoff e^-omega7 we can do","category":"page"},{"location":"","page":"Home","title":"Home","text":"Jc = ExponentialCutoffSD(Johmic, 7)","category":"page"},{"location":"#Defining-custom-spectral-densities","page":"Home","title":"Defining custom spectral densities","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The base AbstractSD can be easily extended by the user. To do so, one must create a new sub-type for which at least one of the methods sd (the spectral density itself J(omega)) or sdoverω (the spectral density divided by frequency J(omega)omega) must be defined. It is strongly recommended to define, if possible, sdoverω since it typically makes it simpler to numerically handle potential singularities.","category":"page"},{"location":"","page":"Home","title":"Home","text":"This is the only requirement, and all other methods will automatically work on the new subtype. Of course, as is typical of Julia's super flexible multiple-dispatch system, the user should consider providing custom functions on the new type for the reorganisation energy, etc.","category":"page"}]
}
