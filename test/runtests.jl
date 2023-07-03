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

    @testset "Weak coupling" begin
        @test WeakCoupling.cauchy_quadgk(x -> 1/(x+2), -1.0, 1.0)[1] ≈ -log(3)/2
        @test WeakCoupling.cauchy_quadgk(x -> x^2, 0.0, 2.0, 1.0)[1] ≈ 4.0
        @test WeakCoupling.hadamard_quadgk(x -> log(x+1), x -> 1/(x+1), 0.0, 2.0, 1.0)[1] ≈ -3*log(9)/4
    end
end