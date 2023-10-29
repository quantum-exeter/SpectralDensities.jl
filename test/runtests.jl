using SpectralDensities
using QuadGK
using ForwardDiff
using Test

@testset "SpectralDensities.jl" begin
    @testset "Reorganisation energies" begin
        Q(J, ωmax=Inf) = quadgk(ω -> sdoverω(J,ω), 0.0, ωmax)[1]

        Jlor = LorentzianSD(rand()*1.5, rand(), rand()/2)
        @test Q(Jlor) ≈ reorganisation_energy(Jlor)
    
        Jdebye = DebyeSD(rand(), rand()*10)
        @test Q(Jdebye) ≈ reorganisation_energy(Jdebye)

        Johmic = OhmicSD(rand()*10)
        Jpoly = PolySD(rand()*10, 7)

        Johmic_hard = HardCutoffSD(Johmic, rand()*20)
        @test Q(Johmic_hard,Johmic_hard.ωcutoff) ≈ reorganisation_energy(Johmic_hard)
        Jpoly_hard = HardCutoffSD(Jpoly, rand()*20)
        @test Q(Jpoly_hard,Jpoly_hard.ωcutoff) ≈ reorganisation_energy(Jpoly_hard)

        Johmic_exp = ExponentialCutoffSD(Johmic, rand()*20)
        @test Q(Johmic_exp) ≈ reorganisation_energy(Johmic_exp) rtol=5e-3
        Jpoly_exp = ExponentialCutoffSD(Jpoly, rand()*20)
        @test Q(Jpoly_exp) ≈ reorganisation_energy(Jpoly_exp) rtol=5e-3

        Johmic_gauss = GaussianCutoffSD(Johmic, rand()*20)
        @test Q(Johmic_gauss) ≈ reorganisation_energy(Johmic_gauss)
        Jpoly_gauss = GaussianCutoffSD(Jpoly, rand()*20)
        @test Q(Jpoly_gauss) ≈ reorganisation_energy(Jpoly_gauss)
    end

    @testset "Singular Integrals" begin
        @test SingularIntegrals.cauchy_quadgk(x -> 1/(x+2), -1.0, 1.0)[1] ≈ -log(3)/2
        @test SingularIntegrals.cauchy_quadgk(x -> x^2, 0.0, 2.0, 1.0)[1] ≈ 4.0
        @test SingularIntegrals.hadamard_quadgk(x -> log(x+1), x -> 1/(x+1), 0.0, 2.0, 1.0)[1] ≈ -3*log(9)/4

        α, ω0, Γ = rand(), 2*rand(), rand()/10
        Jlor = LorentzianSD(α, ω0, Γ)
        ωc = frequency_cutoff(Jlor)
        ωtest = rand(50)*5
        lor_ker_re(ω) = α*(ω0^2 - ω^2)/((ω0^2 - ω^2)^2 + ω^2*Γ^2)
        lor_ker_im(ω) = imag_memory_kernel_ft(Jlor,ω)
        lor_ker_kk(ω) = SingularIntegrals.kramers_kronig(lor_ker_im, ω; cutoff=ωc)
        @test lor_ker_kk.(ωtest) ≈ lor_ker_re.(ωtest)
    end

    @testset "Weak Coupling" begin
        α, ω0, Γ = rand(), 2*rand(), rand()/10
        Jlor = LorentzianSD(α, ω0, Γ)
        T = 0.1
        ωB = 1.0
        ωc = frequency_cutoff(Jlor; tol=1e-8)
        @test isfinite(WeakCoupling.weak_coupling_Σ(Jlor, ωB; ωcutoff=ωc))
        @test isfinite(WeakCoupling.weak_coupling_Σprime(Jlor, ωB; ωcutoff=ωc))
        @test isfinite(WeakCoupling.weak_coupling_Δ(Jlor, ωB, 1/T; ωcutoff=ωc))
        @test_broken isfinite(WeakCoupling.weak_coupling_Δprime(Jlor, ωB, 1/T; ωcutoff=ωc))
    end

    @testset "Memory kernels (time)" begin
        ker(J,τ; ωcutoff=Inf) = quadgk(ω -> 2*J(ω)*sin(ω*τ), 0, ωcutoff)[1] # See AbstractSD

        α, ω0, Γ = rand(), 2*rand(), rand()/10
        Jlor = LorentzianSD(α, ω0, Γ) 
        τtest = LinRange(1e-3, 4, 20)
        modv = memory_kernel.(Jlor, τtest)
        kerv = ker.(Jlor, τtest; ωcutoff=frequency_cutoff(Jlor; tol=1e-8))
        @test isapprox(modv, kerv; rtol=1e-3)

        α, ωc = rand(), 10*rand()
        Jdebye = DebyeSD(α, ωc) 
        τtest = LinRange(1e-2, 1.0, 20)
        modv = memory_kernel.(Jdebye, τtest)
        kerv = ker.(Jdebye, τtest; ωcutoff=frequency_cutoff(Jdebye; tol=1e-8))
        @test_broken maximum(abs.((modv - kerv)./modv)) < 1e-3
    end

    @testset "Memory kernels (frequency)" begin
        α, ω0, Γ = rand(), 2*rand(), rand()/10
        Jlor = LorentzianSD(α, ω0, Γ)
        ωtest = rand(50)*5
        @test imag_memory_kernel_ft.(Jlor,ωtest) ≈ π*Jlor.(ωtest)
        @test imag_memory_kernel_ft.(Jlor,ωtest) ≈ imag.(memory_kernel_ft.(Jlor,ωtest))
        @test real_memory_kernel_ft.(Jlor,ωtest) ≈ real.(memory_kernel_ft.(Jlor,ωtest))

        lor_ker_re(ω) = SingularIntegrals.kramers_kronig(ω -> imag_memory_kernel_ft(Jlor,ω), ω; cutoff=frequency_cutoff(Jlor))
        @test real_memory_kernel_ft.(Jlor,ωtest) ≈ lor_ker_re.(ωtest)
    end

    @testset "InversePolyKernelSD" begin
        α, ω0, Γ = rand(), 2*rand(), rand()/10
        coeffs = [ω0^2/α, -1im*Γ/α, -1/α]
        Jlor = LorentzianSD(α, ω0, Γ)
        Jipk = InversePolyKernelSD(coeffs)
        ωtest = rand(20)
        @test Jlor.(ωtest) ≈ Jipk.(ωtest)
    end

    @testset "Frequency cutoffs" begin
        ωcutoff = 10*rand()
        Johm = OhmicSD(1)
        Jhard = HardCutoffSD(Johm, ωcutoff)
        Jexp = ExponentialCutoffSD(Johm, ωcutoff)
        Jgauss = GaussianCutoffSD(Johm, ωcutoff)
        @test frequency_cutoff(Johm) == Inf
        @test frequency_cutoff(Jhard) == ωcutoff
        @test frequency_cutoff(Jexp; tol=1/ℯ) ≈ ωcutoff
        @test frequency_cutoff(Jgauss; tol=1/ℯ) ≈ ωcutoff

        α, ω0, Γ = rand(), 2*rand(), rand()/10
        Jlor = LorentzianSD(α, ω0, Γ)
        ωc1 = frequency_cutoff(Jlor; tol=1e-3)
        ωc2 = frequency_cutoff(Jlor; tol=1e-5)
        @test ωc1 <= ωc2 && Jlor(ωc1) >= Jlor(ωc2)

        α, ωc = 10*rand(), 20*rand()
        Jdebye = DebyeSD(α, ωc)
        ωmax = ωc
        ωc1 = frequency_cutoff(Jdebye; tol=1e-3)
        ωc2 = frequency_cutoff(Jdebye; tol=1e-5)
        @test ωmax <= ωc1 && ωc1 <= ωc2
        @test Jdebye(ωmax) >= Jdebye(ωc1) && Jdebye(ωc1) >= Jdebye(ωc2)
        @test Jdebye(ωc1)/Jdebye(ωmax) ≈ 1e-3
        @test Jdebye(ωc2)/Jdebye(ωmax) ≈ 1e-5
    end
end