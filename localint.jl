using Games

mutable struct LocalInteraction
	payoff_matrix
	adj_matrix
	players
	current_actions
end

function LocalInteraction(payoff_matrix, adj_matrix, init_actions::Games.PureActionProfile)
	players = [Player(payoff_matrix) for i in 1:size(adj_matrix)[1]]
	current_actions = zeros(Int, size(adj_matrix)[1], size(payoff_matrix)[1])
	for (i, action) in enumerate(init_actions)
		current_actions[i, action] = 1
	end
	return LocalInteraction(payoff_matrix, adj_matrix, players, current_actions)
end

function initialize!(li::LocalInteraction, init_actions::Games.PureActionProfile)
	actions = zeros(Int, size(li.adj_matrix)[1], size(li.payoff_matrix)[1])
	for (i, action) in enumerate(init_actions)
		actions[i, action] = 1
	end
	li.current_actions = actions
	return nothing
end

function play!(li::LocalInteraction, player_ind::Array{T}) where T
	opponent_act_dists = li.adj_matrix[player_ind, :] * li.current_actions
	for (k, i) in enumerate(player_ind)
		best_responses = best_response(li.players[i], opponent_act_dists[k, :])
		action = zeros(T, size(li.payoff_matrix)[1])
		action[best_responses] = 1
		li.current_actions[i, :] = action
	end
	return nothing
end

play!(li::LocalInteraction, player_ind::Int) = play!(li, [player_ind])

play!(li::LocalInteraction) = play!(li, [1:size(li.adj_matrix[1])...])

function get_current_actions(li::LocalInteraction)
	actions = Array{Int}(undef, size(li.adj_matrix)[1])
	for i in 1:size(li.adj_matrix)[1]
		actions[i] = findfirst(x->x==1, li.current_actions[i,:])
	end
	return actions
end