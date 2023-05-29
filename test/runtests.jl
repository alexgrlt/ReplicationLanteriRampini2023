using ReplicationLanteriRampini2023
using Test

@testset "body_replication.jl" 
begin
    # verify that optimal values are close enough to what they should be
    kN_fb, kU_fb, k_fb = ReplicationLanteriRampini2023.run_model()
    @test isapprox(kN_fb[1,1], 16.0846; rtol=1E-2)
    @test isapprox(kU_fb[1,1], 16.0846; rtol=1E-2)
    @test isapprox(kU_fb[1,2], 28.8805; rtol=1E-2)
    @test isapprox(kN_fb[1,2], 28.8805; rtol=1E-2)
end

@testset "test rouwen" begin
    const ρ_s = 0.7 # persistence of schock in AR[1] process modeling idiosyncratic production shock
    const σ_u = 0.12 # standard deviation in AR[1] process 
    const σ_s = σ_u / sqrt(1 - ρ_s^2) # intermediate step to compute the mean of the AR[1] process
    const μ = -0.5 * σ_s^2 # mean of AR[1] process 
    (Z,Π) = ReplicationLanteriRampini2023.rouwen(2,μ, σ_u, ρ_s)
    @test length(Z) == N
    @test length(Π) == N*N
    @test sum(Π, dims=2)[1]
end 