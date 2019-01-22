"""
Filename: test_fictplay.jl
Author: Yuya Furusawa

Tests for fictplay.jl
"""

using Games
using Tests

includes("fictplay.jl")

@testset "Testing fictplay.jl" begin
    
    # Fictitiousplay #

    @testset "Test Two-player Game" begin
        
		matching_pennies_bimatrix = Array{Float64}(undef, 2, 2, 2)
		matching_pennies_bimatrix[:, 1, 1] = [1, -1]
		matching_pennies_bimatrix[:, 1, 2] = [-1, 1]
		matching_pennies_bimatrix[:, 2, 1] = [-1, 1]
		matching_pennies_bimatrix[:, 2, 2] = [1, -1]
		g = NormalFormGame(matching_pennies_bimatrix)
		mp = FictitiousPlay(g)

		belief = get_iterate_result(mp, 1, [1,1])
		for i in 1:2
			@test belief[i][1] == 1
			@test sum(belief[i]) == 1

		@test get_iterate_result(mp, 10, [1,1]) == [[0.4,0.6], [0.3,0.7]]

		assessment_series = get_time_series(mp, 3, [1,1])
		@test assessment_series[1] == [[1.0,0.0], [1.0,0.0]]
		@test assessment_series[2] == [[0.5,0.5], [1.0,0.0]]
		@test assessment_series[3] == [[0.333333, 0.666667], [1.0,1.0]]

    end

end