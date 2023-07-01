using SpectralDensities
using QuadGK
using Test

@testset "SpectralDensities.jl" begin
    @testset "Reorganisation energies" begin
        Q(J) = quadgk(ω -> sdoverω(J,ω), 0.0, Inf)[1]

        Jlor = LorentzianSD(rand()*1.5, rand(), rand()/2)
        @test Q(Jlor) ≈ reorganisation_energy(Jlor)
    
        Jdebye = DebyeSD(rand(), rand()*10)
        @test Q(Jdebye) ≈ reorganisation_energy(Jdebye)

        Johmic = OhmicSD(rand()*10)
        Jpoly = PolySD(rand()*10, 7)

        Johmic_hard = HardCutoffSD(Johmic, rand()*20)
        @test Q(Johmic_hard) ≈ reorganisation_energy(Johmic_hard)
        Jpoly_hard = HardCutoffSD(Jpoly, rand()*20)
        @test Q(Jpoly_hard) ≈ reorganisation_energy(Jpoly_hard)
    end
end