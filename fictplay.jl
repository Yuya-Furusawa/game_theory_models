using Games
using Distributions

abstract type AbstractFictitiousPlay{N} end

mutable struct FictitiousPlay{N} <: AbstractFictitiousPlay{N}
	g::NormalFormGame{N}
	step_size::Function
	x
	t::Int
end

mutable struct StochasticFictitiousPlay{N} <: AbstractFictitiousPlay{N}
	g::NormalFormGame{N}
	step_size::Function
	distribution::Distribution
	x
	t::Int
end

#initialize current_observation and current_period
function FictitiousPlay(g::NormalFormGame{N}, init_actions::Games.MixedActionProfile,
						step_size::Union{Function,Nothing}=nothing) where N
	player_list = [1:N...]
	x = [[init_actions[j] for j in player_list[1:N .!= i]] for i in 1:N]
	if step_size == nothing
		step_size = t -> 1/t
	end
	t = one(Int) #initially period is set to 1
	return FictitiousPlay(g, step_size, tuple(x...), t)
end

function FictitiousPlay(g::NormalFormGame{N}, init_actions::Games.PureActionProfile,
						step_size::Union{Function,Nothing}=nothing) where N
	mixed_actions = [zeros(num_actions(player)) for player in g.players]
	for (i, player) in enumerate(g.players)
		mixed_actions[i] = pure2mixed(num_actions(g.players[i]), init_actions[i])
	end
	return FictitiousPlay(g, tuple(mixed_actions...), step_size)
end

function StochasticFictitiousPlay(g::NormalFormGame{N}, distribution::Distribution,
								  init_actions::Games.MixedActionProfile, step_size::Union{Function,Nothing}=nothing) where N
	player_list = [1:N...]
	x = [[init_actions[j] for j in player_list[1:N .!= i]] for i in 1:N]
	if step_size == nothing
		step_size = t -> 1/t
	end
	t = one(Int) #initially period is set to 1
	return StochasticFictitiousPlay(g, step_size, distribution, tuple(x...), t)
end

function StochasticFictitiousPlay(g::NormalFormGame{N}, distribution::Distribution,
								  init_actions::Games.PureActionProfile, step_size::Union{Function,Nothing}=nothing) where N
	mixed_actions = [zeros(num_actions(player)) for player in g.players]
	for (i, player) in enumerate(g.players)
		mixed_actions[i] = pure2mixed(num_actions(g.players[i]), init_actions[i])
	end
	return StochasticFictitiousPlay(g, distribution, tuple(mixed_actions...), step_size)
end

#Intitialize observations and periods
function initialize!(afp::AbstractFictitiousPlay{N}, init_actions::Games.MixedActionProfile) where N
	player_list = [1:N...]
	x = [[init_actions[j] for j in player_list[1:N .!= i]] for i in 1:N]
	afp.x = tuple(x...)
	afp.t = one(Int)
	return afp.x
end

function initialize!(afp::AbstractFictitiousPlay{N}, init_actions::Games.PureActionProfile) where N
	mixed_actions = [zeros(num_actions(player)) for player in afp.g.players]
	for (i, player) in enumerate(afp.g.players)
		mixed_actions[i] = pure2mixed(num_actions(afp.g.players[i]), init_actions[i])
	end
	initialize!(afp, tuple(mixed_actions...))
end

#preceed one round with updaing a state
function play!(afp::AbstractFictitiousPlay{N}) where N
	player_list = [1:N...]
	actions = zeros(Int, N)
	if isa(afp, FictitiousPlay)
		payoff_pertubation_dist = size -> zeros(size)
	else
		payoff_pertubation_dist = size -> rand(afp.distribution, size)
	end
	payoff_pertubation = Vector{Any}(undef, N)
	for (i, player) in enumerate(afp.g.players)
		payoff_pertubation[i] = payoff_pertubation_dist(num_actions(afp.g.players[i]))
		actions[i] = best_response(player, tuple(afp.x[i]...), payoff_pertubation[i])
	end
	for (i, player) in enumerate(afp.g.players)
		for j in player_list[1:N .!= i]
			k = j > i ? j-1 : j
			afp.x[i][k] *= (1 - afp.step_size(afp.t+1))
			afp.x[i][k][actions[j]] += afp.step_size(afp.t+1)
		end
	end
	afp.t = afp.t + 1
	return afp.x
end

function play!(afp:: AbstractFictitiousPlay{N}, num_iter::Int) where N
	for t in num_iter
		play!(afp)
	end
end

function play!(afp::AbstractFictitiousPlay{N}, init_actions::ActionProfile, num_iter::Int) where N
	initialize!(afp, init_actions)
	for t in 1:num_iter
		play!(afp)
	end
end