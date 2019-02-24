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
	x
	t::Int
	distribution::Distribution
end

#initialize current_observation and current_period
function FictitiousPlay(g::NormalFormGame{N}, init_actions::Games.MixedActionProfile,
						step_size::Union{Function,Nothing}=nothing) where N
	if step_size == nothing
		step_size = t -> 1/t
	end
	t = one(Int) #initially period is set to 1
	return FictitiousPlay(g, step_size, [init_actions...], t)
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
	if step_size == nothing
		step_size = t -> 1/t
	end
	t = one(Int) #initially period is set to 1
	return StochasticFictitiousPlay(g, step_size, [init_actions...], t, distribution)
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
	afp.x = [init_actions...]
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
	actions = zeros(Int, N)
	if isa(afp, FictitiousPlay)
		payoff_perturbation_dist = size -> zeros(size)
	else
		payoff_perturbation_dist = size -> rand(afp.distribution, size)
	end
	payoff_perturbation = Vector{Any}(undef, N)
	for (i, player) in enumerate(afp.g.players)
		opponent_actions = [afp.x[j] for j in 1:N if j != i]
		payoff_perturbation[i] = payoff_perturbation_dist(num_actions(afp.g.players[i]))
		actions[i] = best_response(player, tuple(opponent_actions...), payoff_perturbation[i])
	end
	for (i, player) in enumerate(afp.g.players)
		afp.x[i] *= (1 - afp.step_size(afp.t+1))
		afp.x[i][actions[i]] += afp.step_size(afp.t+1)
	end
	afp.t = afp.t + 1
	return afp.x
end

function play!(afp:: AbstractFictitiousPlay{N}, num_iter::Int) where N
	for t in num_iter
		play!(afp)
	end
	return afp.x
end

function play!(afp::AbstractFictitiousPlay{N}, init_actions::ActionProfile, num_iter::Int) where N
	initialize!(afp, init_actions)
	for t in 1:num_iter
		play!(afp)
	end
	return adp.x
end