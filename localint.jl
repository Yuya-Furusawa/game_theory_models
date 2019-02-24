using Games

mutable struct LocalInteraction
	N::Int
	m::Int
	adj_matrix
	players
	current_actions
end

function LocalInteraction(payoff_matrix, adj_matrix, init_actions::Games.PureActionProfile)
	N = size(adj_matrix)[1]
	m = size(payoff_matrix)[1]
	players = [Player(payoff_matrix) for i in 1:N]
	current_actions = zeros(Int, N, m)
	for (i, action) in enumerate(init_actions)
		current_actions[i, action] = 1
	end
	return LocalInteraction(N, m, adj_matrix, players, current_actions)
end

function LocalInteraction(players::NTuple{N,Player{N}}, adj_matrix, init_actions::Games.PureActionProfile) where N
	if size(adj_matrix)[1] != N
		throw(ArgumentError("The number of players must be equivalent to the number of the rows of adjacency matrix"))
	end
	for i in 1:N-1
		num_actions(players[i]) != num_actions(players[i+1]) && throw(ArgumentError("The number of actions must be same across all players"))
	end
	m = num_actions(players[1])
	players = [players[i] for i in 1:N]
	current_actions = zeros(Int, N, m)
	for (i, action) in enumerate(init_actions)
		current_actions[i, action] = 1
	end
	return LocalInteraction(N, m, adj_matrix, players, current_actions)
end


function initialize!(li::LocalInteraction, init_actions::Games.PureActionProfile)
	actions = zeros(Int, li.N, li.m)
	for (i, action) in enumerate(init_actions)
		actions[i, action] = 1
	end
	li.current_actions = actions
	return li.current_actions
end

function play!(li::LocalInteraction, player_ind::Vector{T}) where T
	opponent_act_dists = li.adj_matrix[player_ind, :] * li.current_actions
	for (k, i) in enumerate(player_ind)
		best_responses = best_response(li.players[i], opponent_act_dists[k, :])
		action = zeros(T, li.m)
		action[best_responses] = 1
		li.current_actions[i, :] = action
	end
	return li.current_actions
end

play!(li::LocalInteraction, player_ind::Int) = play!(li, [player_ind])

play!(li::LocalInteraction) = play!(li, [1:li.N...])

function play!(li::LocalInteraction, player_ind::Vector{Int}, num_iter::Int)
	for t in 1:num_iter
		play!(li, player_ind)
	end
	return li.current_actions
end

play!(li::LocalInteraction, player_ind::Int, num_iter::Int) = play!(li, [player_ind], num_iter)

function get_current_actions(li::LocalInteraction)
	actions = Array{Int}(undef, li.N)
	for i in 1:li.N
		actions[i] = findfirst(x->x==1, li.current_actions[i,:])
	end
	return actions
end