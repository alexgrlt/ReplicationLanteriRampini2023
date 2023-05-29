using ReplicationLanteriRampini2023
using Test

@testset "ReplicationLanteriRampini2023.jl" begin

    # verify that optimal values are close enough to what they should be
    isapprox(kN_fb[1,1], 16.0846; rtol=1E-3)
    isapprox(kU_fb[1,1], 16.0846; rtol=1E-3)
    isapprox(kU_fb[1,2], 28.8805; rtol=1E-3)
    isapprox(kN_fb[1,2], 28.8805; rtol=1E-3)
end

@testset "rouwen.jl"
begin 
@test length(Z) == N
@test length(Π) == N*N
@test sum(Π, dims=2)[1]
end 