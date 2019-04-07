# ------------------------------- #
# Testing best response dynamics  #
# ------------------------------- #

using Games
using Test

include("brd.jl")


@testset "Testing brd.jl" begin
    
    payoff_matrix = [4 0; 3 2]
    N = 4
    ts_length = 3
    init_actions = (1,1,1,1)

    @testset "Testing best response dynamics model" begin
        
        brd = BRD(N, payoff_matrix)
        @test @inferred(time_series(brd, ts_length, init_actions)) ==
              [4 4 4; 0 0 0]
    end

    @testset "Testing KMR model" begin
        
        epsilon = 0.1
        kmr = KMR(N, payoff_matrix, epsilon)
        series = time_series(kmr, ts_length, init_actions)
        for t in 1:3
            @test sum(series[:, t]) == 4
        end
    end

    @testset "Testing sampling best response dynamics model" begin
        
        k = 2
        sbrd = SamplingBRD(N, payoff_matrix, k)
        series = time_series(sbrd, ts_length, init_actions)
        for t in 1:3
            @test sum(series[:, t]) == 4
        end
    end
end