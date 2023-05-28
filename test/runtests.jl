using ReplicationLanteriRampini2023
using Test

@testset "ReplicationLanteriRampini2023.jl" begin
    # Write your tests here.
end

@testset "rouwen.jl"
begin 
@test length(Z) == N
@test length(Π) == N*N
@test sum(Π, dims=2)[1]
end 