"""
Filename: fictplay.py

Authors: Daisuke Oyama

Fictitious play model.

"""
from __future__ import division

import numpy as np
import copy
from normal_form_game import NormalFormGame, pure2mixed, Player


class FictitiousPlay(object):
    """
    Class representing the Fictitious play model with two players

    Parameters
    ----------
    data : array_like(float) or `NormalFormGame` object
        The payoff matrix of the two-player game

    Attributes
    ----------
    g : `NormalFormGame` object

    N : scalar(int)
        Number of players

    players : tuple(Player)
        tuple of Player instances

    nums_actions : tuple(int)
        Number of actions for each player

    """

    def __init__(self, data):
        if isinstance(data, NormalFormGame):
            self.g = data
        else:  # data must be array_like
            payoffs = np.asarray(data)
            self.g = NormalFormGame(payoffs)

        self.N = self.g.N 
        self.players = self.g.players
        self.nums_actions = self.g.nums_actions
        self.tie_breaking = 'smallest'

        self.current_actions = np.zeros(self.N, dtype=int)

        # Create the list of opponents for each player
        self.opponent_list = list(list(range(self.N)) for i in range(self.N))
        for i in range(self.N):
            self.opponent_list[i].remove(i)
        
        # Create instance variable `norm_action_hist` for players
        for i, player in enumerate(self.players):
            player.norm_action_hist = np.empty(self.nums_actions[i])

        self._initial_weight = lambda t: 1 / (t+1)
        self.step_size = self._initial_weight

    def __repr__(self):
        msg = "Fictitious play for "
        g_repr = self.g.__repr__()
        msg += g_repr
        return msg

    def __str__(self):
        return self.__repr__()

    @property
    def norm_action_hist(self):
        return tuple(player.norm_action_hist for player in self.players)

    def set_init_actions(self, init_actions=None):
        """
        Randomly sets `current_actions` if `init_actions` is None.
        And initialize `norm_action_hist` for each player given `current_actions`
        """
        if init_actions is None:
            init_actions = np.zeros(self.N, dtype=int)
            for i, n in enumerate(self.nums_actions):
                init_actions[i] = np.random.randint(n)
        self.current_actions[:] = init_actions

        for i, player in enumerate(self.players):
            player.norm_action_hist[:] = pure2mixed(self.nums_actions[i], init_actions[i])

    def play(self):
        """
        The method used to proceed the game by one period. Players take their
        best response strategies given noramlized action history about opponents actions.

        """
        for i, player in enumerate(self.players):
            opponent_actions = np.asarray([opponent.norm_action_hist for opponent in self.players if opponent != player])
            self.current_actions[i] = \
                player.best_response(opponent_actions if self.N > 2 else opponent_actions[0],
                                    tie_breaking=self.tie_breaking)

    def update_norm_action_hist(self, step_size):
        """
        This method used to update each players' normalized action history. Normalized action histories are updated by
        opponent's action and `step_size`.

        """
        for i, player in enumerate(self.players):
            player.norm_action_hist[:] *= 1 - step_size
            player.norm_action_hist[self.current_actions[i]] += step_size

    def _sum_index(self, i):
        if i == 0:
            return 0
        else:
            return sum(self.nums_actions[j] for j in range(i))

    def get_time_series(self, ts_length, init_actions=None):
        """
        Return the noramlized action history of each players in each round.

        Parameters
        ----------
        ts_length : scalar(int)
            The number of rounds you play in a simulation.

        """
        norm_action_hist_sequence = np.empty((ts_length, sum(self.nums_actions)))
        norm_action_hist_iter = self.get_time_series_iter(ts_length, init_actions)

        for t, norm_action_hist in enumerate(norm_action_hist_iter):
            for i in range(self.N):
                norm_action_hist_sequence[t, self._sum_index(i):self._sum_index(i+1)] = norm_action_hist[i]

        norm_action_hist_list = []
        for i in range(self.N):
            norm_action_hist_list.append(norm_action_hist_sequence[:, self._sum_index(i):self._sum_index(i+1)])

        return tuple(norm_action_hist_list)

    def get_time_series_iter(self, ts_length, init_actions=None):
        """
        Iterater version of `get_time_series` method

        Parameters
        ----------
        ts_length : scalar(int)
            The number of rounds you play in a simulation.

        """
        self.set_init_actions(init_actions)

        for t in range(ts_length):
            yield self.norm_action_hist
            self.play()
            self.update_norm_action_hist(self.step_size(t+1))

    def iterate_result(self, ts_length, init_actions=None):
        """
        Returns the ultimate normalized action history of each players after `ts_length` times play.

        Parameters
        ----------
        ts_length : scalar(int)
            The number of periods you play in a simulation.

        """
        self.set_init_actions(init_actions)

        for t in range(ts_length):
            self.play()
            self.update_asseccements(self.step_size(t+1))
        return self.norm_action_hist


class StochasticFictitiousPlay(FictitiousPlay):
    """
    Stochastic fictitious play with two players.

    Parameters
    ----------
    data : array_like(float) or `NormalFormGame` object
        the payoff matrix of two-player game

    distribution : {'extreme', 'normal'}
        optional(default='extreme')

    epsilon : scalar(float)
        optional(default=None)

    """

    def __init__(self, data, distribution='extreme', epsilon=None):
        FictitiousPlay.__init__(self, data)

        if distribution == 'extreme':  # extreme-value, or gumbel, distribution
            loc = -np.euler_gamma * np.sqrt(6) / np.pi
            scale = np.sqrt(6) / np.pi
            self.payoff_perturbation_dist = \
                lambda size: np.random.gumbel(loc=loc, scale=scale, size=size)
        elif distribution == 'normal':  # standard normal distribution
            self.payoff_perturbation_dist = \
                lambda size: np.random.standard_normal(size=size)
        else:
            raise ValueError("`distribution` must be 'extreme' or 'normal'")

        self._epsilon = epsilon

        self.tie_breaking = 'smallest'

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        self._epsilon = value
        self._set_step_size()

    def _set_step_size(self):
        # Set step size:
        # If epsilon is None, step_size = _decreasing_gain,
        # otherwise, step_size = epsilon
        if self._epsilon is None:
            self.step_size = self._initial_weight
        else:
            self.step_size = lambda t: self._epsilon

    def play(self):
        """
        The method used to proceed the game by one period. Players take their
        best response strategies given assessments about opponents actions.
        """
        n = sum(self.nums_actions)
        random_values = self.payoff_perturbation_dist(size=n)
        payoff_perturbations = []
        for i in range(self.N):
            payoff_perturbations.append(random_values[self._sum_index(i):self._sum_index(i+1)])

        for i, player in enumerate(self.players):
            opponent_actions = np.asarray([opponent.norm_action_hist for opponent in self.players if opponent != player])
            self.current_actions[i] = player.best_response(
                opponent_actions if self.N > 2 else opponent_actions[0],
                tie_breaking=self.tie_breaking,
                payoff_perturbation=payoff_perturbations[i]
            )