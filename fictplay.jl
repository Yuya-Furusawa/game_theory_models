#=
Tools for fictitious play models

Author: Yuya Furusawa
=#


using Games
using Distributions


#AbstractFictitiousPlay#

"""
	AbstractFictitiousPlay{N}

Abstract type representing fictitious play and stochastic fictitious play models.
"""

abstract type AbstractFictitiousPlay{N} end 

#FictitiousPlay#

"""
	FictitiousPlay{N}

Type representing a fictitious play model. The subtype of `AbstractFictitiousPlay`.

#Fields

- `game::NormalFormGame{N}` : Normal form game used in the fictitious play model.
"""

struct FictitiousPlay{N} <: AbstractFictitiousPlay{N}
	game::NormalFormGame{N}
end

#StochasticFictitiousPlay#

"""
	StochasticFictitiousPlay{N}

Type representing a stochastic fictitious play model.
The subtype of `AbstractFictitiousPlay`.

#Fields

- `game::NormalFormGame{N}` : NormalFormGame form game used in the stochastic
							fictitious play model.
- `distribution::Symbol` : The distribution of payoff shocks i in each rounds.
							Must be `:extreme` or `:normal`.
"""

struct StochasticFictitiousPlay{N} <: AbstractFictitiousPlay{N}
	game::NormalFormGame{N}
	distribution::Symbol
end


# _set_actions

"""
	_set_actions(g, init_actions)

Set each player's actions specified by `init_actions`. If `init_actions` is
`nothing`, it randomly sets the actions.

# Arguments

- `g::AbstractFictitiousPlay` : `FictitiousPlay` or `StochasticFictitiousPlay` instance.
- `init_actions::Union{ActionProfile,Nothing}` : Actions to be set.
"""

function _set_actions(g::AbstractFictitiousPlay{N},
					init_actions::Union{ActionProfile,Nothing}) where N
	#Set Actions
	if init_actions == nothing
		init_actions = zeros(Int, N)
		for (i, player) in enumerate(g.game.players)
			init_actions[i] = rand(1:num_actions(player))
		end
	end
	return init_actions
end

# _set_assessments

"""
	_set_assessments(g, init_actions)

Set each player's assessments specified by `init_actions`.

# Arguments

- `g::AbstractFictitiousPlay` : `FictitiousPlay` or `StochasticFictitiousPlay` instance.
- `init_actions::ActionProfile` : Actions used to set assessments.
"""

function _set_assessments(g::AbstractFictitiousPlay{N},
						init_actions::ActionProfile) where N
	#Set Beliefs
	player_list = [1:N...]
	assessments =
		[[zeros(num_actions(g.game.players[j])) for j in player_list[1:N .!= i]] for i in 1:N]
	for (i, player) in enumerate(g.game.players)
		for j in player_list[1:N .!= i]
			k = j > i ? j-1 : j
			assessments[i][k] =
				pure2mixed(num_actions(g.game.players[j]), init_actions[j])
		end
	end
	return assessments
end


# get_iterate_result

"""
	get_iterate_result(g, ts_length[, init_actions])

Return the each player's assessments after `ts_length` times iteration.

# Arguments

- `g::FictitiousPlay` : `FictitiousPlay` instance.
- `ts_length::Int` : The number of periods you play.
- `init_actions::Union{ActionProfile,Nothing}` : Actions designated as initial actions.

# Returns

- `::Array` : The assessments of players.
"""

function get_iterate_result(g::FictitiousPlay{N}, ts_length::Int,
						init_actions::Union{ActionProfile,Nothing}=nothing) where N
	#Set Actions
	init_actions = _set_actions(g,init_actions)
	#Set Beliefs
	assessments = _set_assessments(g,init_actions)
	#Body
	for t in 1:ts_length
		#play
		for (i, player) in enumerate(g.game.players)
			init_actions[i] = best_response(player, tuple(assessments[i]...))
		end
		#update_assessments
		player_list = [1:N...]
		for (i, player) in enumerate(g.game.players)
			for j in player_list[1:N .!= i]
				k = j > i ? j-1 : j
				assessments[i][k] = assessments[i][k] * (1 - 1/(t+1))
				assessments[i][k][init_actions[j]] =
									assessments[i][k][init_actions[j]] + 1/(t+1)
			end
		end
	end
	#Return
	return [assessments[i][1] for i in 1:N] #Array of Array
end

"""
	get_iterate_result(g, ts_length[, init_actions, epsilon])

Return the each player's assessments after `ts_length` times iteration.

# Arguments

- `g::StochasticFictitiousPlay` : `StochasticFictitiousPlay` instance.
- `ts_length::Int` : The number of periods you play.
- `init_actions::Union{ActionProfile,Nothing}` : Actions designated as initial actions.
- `epsilon::Union{<:Real,Nothing}` : Weights used on updating assessments.

# Returns

- `::Array` : The assessments of players.
"""

function get_iterate_result(g::StochasticFictitiousPlay{N}, ts_length::Int,
							init_actions::Union{ActionProfile,Nothing}=nothing,
							epsilon::Union{<:Real,Nothing}=nothing) where N
	#Set Actions
	init_actions = _set_actions(g,init_actions)
	#Set Beliefs
	assessments = _set_assessments(g,init_actions)
	#Set Distribution
	if 	g.distribution == :extreme
		loc = MathConstants.eulergamma * sqrt(6) / pi
		scale = sqrt(6) / pi
		gumbel = Gumbel(loc, scale)
		payoff_pertubation_dist = size -> rand(gumbel, size)
	elseif g.distribution == :normal
		normal = Normal()
		payoff_pertubation_dist = size -> rand(normal, size)
	else
		throw(ArgumentError("`distribution must be `extreme` or `normal`"))
	end
	#Set Step Size
	if epsilon == nothing
		step_size = t -> t
	else 
		step_size = t -> epsilon
	end
	#Body
	for t in 1:ts_length
		#play
		payoff_pertubation = Vector{Any}(undef, N)
		for (i, player) in enumerate(g.game.players)
			payoff_pertubation[i] =
				payoff_pertubation_dist(num_actions(g.game.players[i]))
			init_actions[i] =
				best_response(player, tuple(assessments[i]...), payoff_pertubation[i])
		end
		#update_assessments
		player_list = [1:N...]
		for (i, player) in enumerate(g.game.players)
			for j in player_list[1:N .!= i]
				k = j > i ? j-1 : j
				assessments[i][k] = assessments[i][k] * (1 - 1/step_size(t+1))
				assessments[i][k][init_actions[j]] =
						assessments[i][k][init_actions[j]] + 1/step_size(t+1)
			end
		end
	end
	#Return
	return [assessments[i][1] for i in 1:N] #Array of Array
end

# get_time_series

"""
	get_time_series(g, ts_length[, init_actions])

Return the array of the sequences of each player's assessments.

# Arguments

- `g::FictitiousPlay` : `FictitiousPlay` instance.
- `ts_length::Int` : The number of periods you play.
- `init_actions::Union{ActionProfile,Nothing}` : Actions designated as initial actions.

# Returns

- `::Array` : The sequence of players' assessments in each rounds.
"""

function get_time_series(g::FictitiousPlay{N}, ts_length::Int,
					init_actions::Union{ActionProfile,Nothing}=nothing) where N
	#Set Actions
	init_actions = _set_actions(g,init_actions)
	#Set Beliefs
	assessments = _set_assessments(g,init_actions)
	#Create assessment sequence
	assessment_sequences = Array{Any}(undef, ts_length, N)
	#Iteration
	for t in 1:ts_length
		#Substitution
		assessment_sequences[t,:] = [assessments[i][1] for i in 1:N]
		#play
		for (i, player) in enumerate(g.game.players)
			init_actions[i] = best_response(player, tuple(assessments[i]...))
		end
		#update_assessments
		player_list = [1:N...]
		for (i, player) in enumerate(g.game.players)
			for j in player_list[1:N .!= i]
				k = j > i ? j-1 : j
				assessments[i][k] = assessments[i][k] * (1 - 1/(t+1))
				assessments[i][k][init_actions[j]] =
								assessments[i][k][init_actions[j]] + 1/(t+1)
			end
		end
	end
	#Return
	return [assessment_sequences[:,i] for i in 1:N] #ts_length x N Array
end

"""
	get_time_series(g, ts_length[, init_actions, epsilon])

Return the array of the sequences of each player's assessments.

# Arguments

- `g::StochasticFictitiousPlay` : `StochasticFictitiousPlay` instance.
- `ts_length::Int` : The number of periods you play.
- `init_actions::Union{ActionProfile,Nothing}` : Actions designated as initial actions.
- `epsilon::{<:Real,Nothing}` : Weights used on updating assessments.

# Returns

- `::Array` : The sequence of players' assessments in each rounds.
"""


function get_time_series(g::StochasticFictitiousPlay{N}, ts_length::Int,
						init_actions::Union{ActionProfile,Nothing}=nothing,
						epsilon::Union{<:Real,Nothing}=nothing) where N
	#Set Actions
	init_actions = _set_actions(g,init_actions)
	#Set Beliefs
	assessments = _set_assessments(g,init_actions)
	#Set Distribution
	if g.distribution == :extreme
		loc = MathConstants.eulergamma * sqrt(6) / pi
		scale = sqrt(6) / pi
		gumbel = Gumbel(loc, scale)
		payoff_pertubation_dist = size -> rand(gumbel, size)
	elseif g.distibution == :normal
		normal = Normal()
		payoff_pertubation_dist = size -> rand(normal, size)
	else
		throw(ArgumentError("`distribution must be `extreme` or `normal`"))
	end
	#Set Step Size
	if epsilon == nothing
		step_size = t -> t
	else 
		step_size = t -> epsilon
	end
	#Create assessment sequence
	assessment_sequences = Array{Any}(undef, ts_length, N)
	#Iteration
	for t in 1:ts_length
		#Substitution
		assessment_sequences[t,:] = [assessments[i][1] for i in 1:N]
		#play
		payoff_pertubation = Vector{Any}(undef, N)
		for (i, player) in enumerate(g.game.players)
			payoff_pertubation[i] =
				payoff_pertubation_dist(num_actions(g.game.players[i]))
			init_actions[i] =
				best_response(player, tuple(assessments[i]...), payoff_pertubation[i])
		end
		#update_assessments
		player_list = [1:N...]
		for (i, player) in enumerate(g.game.players)
			for j in player_list[1:N .!= i]
				k = j > i ? j-1 : j
				assessments[i][k] = assessments[i][k] * (1 - 1/step_size(t+1))
				assessments[i][k][init_actions[j]] =
						assessments[i][k][init_actions[j]] + 1/step_size(t+1)
			end
		end
	end
	#Return
	return [assessment_sequences[:,i] for i in 1:N] #ts_length x N Array
end