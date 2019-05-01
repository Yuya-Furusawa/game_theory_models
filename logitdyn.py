"""
Filename: logitdyn.py

Authors: Daisuke Oyama

Logit dynamics.

"""
import numpy as np
import numbers
from util import check_random_state
from normal_form_game import *
from random_game import random_pure_actions


class LogitDynamics(object):
    """
    Parameters
    ----------
    g : NormalFormGame
        N-player game played

    beta : scalar(float)

    Attributes
    ----------
    N : scalar(int)
        The number of players in the game

    players : list(Player)
        The list consisting of all players with the given payoff matrix.

    nums_actions : tuple(int)
        Tuple of the number of actions, one for each player.

    beta : scalar(float)
        See Parameters.

    player.logit_choice_cdfs : array-like
        The choice probability of each actions given opponents' actions.
    """
    def __init__(self, g, beta=1.0):
        self.N = g.N
        self.players = g.players
        self.nums_actions = g.nums_actions

        self.beta = beta

        for player in self.players:
            payoff_array_rotated = \
                player.payoff_array.transpose(list(range(1, self.N)) + [0])
            # Shift payoffs so that max = 0 for each opponent action profile
            payoff_array_rotated -= \
                payoff_array_rotated.max(axis=-1)[..., np.newaxis]
            # cdfs left unnormalized
            player.logit_choice_cdfs = \
                np.exp(payoff_array_rotated*self.beta).cumsum(axis=-1)
            # player.logit_choice_cdfs /= player.logit_choice_cdfs[..., [-1]]

    def logit_choice_cdfs(self):
        return tuple(player.logit_choice_cdfs for player in self.players)

    def _play(self, player_ind, actions, random_state=None):
        random_state = check_random_state(random_state)
        i = player_ind

        # Tuple of the actions of opponent players i+1, ..., N, 0, ..., i-1
        opponent_actions = \
            tuple(actions[i+1:]) + tuple(actions[:i])

        cdf = self.players[i].logit_choice_cdfs[opponent_actions]
        random_value = random_state.rand()
        next_action = cdf.searchsorted(random_value*cdf[-1], side='right')

        return next_action

    def play(self, init_actions=None, player_ind_seq=None, num_reps=1,
             random_state=None):
        random_state = check_random_state(random_state)
        if init_actions is None:
            init_actions = random_pure_actions(self.nums_actions, random_state)
        init_actions = list(init_actions)

        if player_ind_seq is None:
            player_ind_seq = random_state.randint(self.N, size=num_reps)

        if isinstance(player_ind_seq, numbers.Integral):
            player_ind_seq = [player_ind_seq]

        for t, player_ind in enumerate(player_ind_seq):
            init_actions[player_ind] = self._play(player_ind, init_actions,
                                                  random_state)

        return tuple(init_actions)

    def time_series(self, ts_length, init_actions=None, random_state=None):
        random_state = check_random_state(random_state)
        if init_actions is None:
            init_actions = random_pure_actions(self.nums_actions, random_state)
        actions = [action for action in init_actions]

        out = np.empty((ts_length, self.N), dtype=int)
        player_ind_seq = random_state.randint(self.N, size=ts_length)

        for t, player_ind in enumerate(player_ind_seq):
            for i in range(self.N):
                out[t, i] = actions[i]
            actions[player_ind] = self._play(player_ind, actions, random_state)

        return out
