#=
    Tools for Logit Response Dynamics
=#

using Games
using Random


# LogitDynamics #

"""
    LogitDynamics{N, T, S}

Type representing Logit-Dynamics model.

# Fields

- `players::NTuple{N,Player{N,T}}` : Tuple of player instances.
- `nums_actions::NTuple{N,Int}` : Tuple of integers which are the number of
    actions for each player.
- `beta::S` : The level of noise in players' decision.
- `choice_probs` : Each players' choice probabilities for each actions.
"""
struct LogitDynamics{N,T<:Real,S<:Real}
    players::NTuple{N,Player{N,T}}
    nums_actions::NTuple{N,Int}
    beta::S
    choice_probs
end

"""
    LogitDynamics(g, beta)

Construct a `LogitDynamics` instance.

# Arguments

- `g::NormalFormGame{N,T}` : `NormalFormGame` instance.
- `beta::S` : The level of noise in players' decision.

# Return

- `::LogitDynamics` : New `LogitDynamics` instance.
"""
function LogitDynamics(g::NormalFormGame{N,T}, beta::S) where {N,T<:Real,S<:Real}
    choice_probs = Vector{Any}(undef, N)
    for (i, player) in enumerate(g.players)
        payoff_array = player.payoff_array
        payoff_array_normalized = payoff_array .- maximum(payoff_array, dims=N)
        choice_probs[i] = cumsum(exp.(payoff_array_normalized .* beta),
                                       dims=N)
    end
    return LogitDynamics(g.players, g.nums_actions, beta, choice_probs)
end

"""
    play!([rng,] lgdy, player_ind, actions)

Return new action of player indexed by `player_ind` given each players' choice
probabilities.

# Arguments

- `rng::AbstractRNG` : Random number generator used.
- `lgdy::LogitDynamics{N}` : `LogitDynamics` instance.
- `player_ind::Integer` : Player index who takes action.
- `actions::Vector{<:Integer}` : Vector of actions for each players.

# Return

- `::Integer` : The new action of player indexed by `player_ind`.
"""
function play!(rng::AbstractRNG, lgdy::LogitDynamics{N}, player_ind::Integer,
               actions::Vector{<:Integer}) where N
    oppponent_actions = [actions[player_ind+1:N]..., actions[1:player_ind-1]...]
    cdf = lgdy.choice_probs[player_ind][oppponent_actions..., :]
    random_value = rand(rng)
    next_action = searchsortedfirst(cdf, random_value*cdf[end])
    return next_action
end

"""
    play([rng,] lgdy, init_actions; num_reps)

Return new action profile after `num_reps` iterations.

# Arguments

- `rng::AbstractRNG` : Random number generator used.
- `lgdy::LogitDynamics{N}` : `LogitDynamics` instance.
- `init_actions::Games.PureActionProfile` : Initial action profile.
- `num_reps::Integer` : The number of iterations; defaults to 1.

# Return

- `::Vector{<:Integer}` : New action profile.
"""
function play(rng::AbstractRNG,
              lgdy::LogitDynamics{N},
              init_actions::Games.PureActionProfile;
              num_reps::Integer=1) where N
    actions = [m for m in init_actions]
    player_ind_seq = rand(rng, 1:N, num_reps)
    for player_ind in player_ind_seq
        actions[player_ind] = play!(rng, lgdy, player_ind, actions)
    end
    return actions
end

play(lgdy::LogitDynamics, init_actions::Games.PureActionProfile;
     num_reps::Integer=1) =
    play(Random.GLOBAL_RNG, lgdy, init_actions, num_reps=num_reps)

"""
    time_series!(rng, lgdy, out, player_ind_seq)

Update `out` which represents the time series of action profile.

# Arguments

- `rng::AbstractRNG` : Random number generator used.
- `lgdy::LogitDynamics{N}` : `LogitDynamics` instance.
- `out::Matrix{<:Integer}` : Matrix which represents the time series of action
    profile.
- `player_ind_seq::Vector{<:Integer}` : The sequence of player index, which is
    determined randomly.

# Return

- `::Matrix{<:Integer}` : Updated `out`.
"""
function time_series!(rng::AbstractRNG,
                      lgdy::LogitDynamics{N},
                      out::Matrix{<:Integer},
                      player_ind_seq::Vector{<:Integer}) where N
    ts_length = size(out, 2)
    current_actions = [out[i, 1] for i in 1:N]
    for t in 1:ts_length-1
        current_actions[player_ind_seq[t]] = play!(rng, lgdy, player_ind_seq[t],
                                                   current_actions)
        for i in 1:N
            out[i, t+1] = current_actions[i]
        end
    end
    return out
end

"""
    time_series([rng,] lgdy, ts_length, init_actions)

Return the time series of action profile

# Arguments

- `rng::AbstractRNG` : Random number generator used.
- `lgdy::LogitDynamics{N}` : `LogitDynamics` instance.
- `ts_length::Integer` : The length of time series.
- `init_actions::Games.PureActionProfile` : Initial action profile.
"""
function time_series(rng::AbstractRNG,
                     lgdy::LogitDynamics{N},
                     ts_length::Integer,
                     init_actions::Games.PureActionProfile) where N
    player_ind_seq = rand(rng, 1:N, ts_length-1)
    out = Matrix{Int}(undef, N, ts_length)
    for i in 1:N
        out[i, 1] = init_actions[i]
    end
    time_series!(rng, lgdy, out, player_ind_seq)
end

time_series(lgdy::LogitDynamics, ts_length::Integer,
            init_actions::Games.PureActionProfile) =
    time_series(Random.GLOBAL_RNG, lgdy, ts_length, init_actions)
