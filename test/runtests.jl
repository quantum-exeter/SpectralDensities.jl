using SpectralDensities
using QuadGK
using Test

@testset "SpectralDensities.jl" begin
    @testset "Reorganisation energies" begin
        Q(J) = quadgk(ω -> sdoverω(J,ω), 0.0, Inf)[1]

        Jlor = LorentzianSD(rand()*1.5, rand(), rand()/2)
        @test Q(Jlor) ≈ reorganisation_energy(Jlor)
    end
end