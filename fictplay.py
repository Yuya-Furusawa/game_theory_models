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

        self.assessment_sizes = tuple(
                tuple(self.nums_actions[j] for j in self.opponent_list[i]) for i in range(self.N)
                )
        
        # Create instance variable `current_assessment` for self.players
        for i, player in enumerate(self.players):
            player.current_assessment = np.asarray([np.empty(self.nums_actions[j]) for j in self.opponent_list[i]])

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
    def current_assessments(self):
        return tuple(player.current_assessment for player in self.players)

    def set_init_actions(self, init_actions=None):
        """
        Randomly sets `current_actions` if `init_actions` is None.
        And initialize `current_assessment` for each player given `current_actions`
        """
        if init_actions is None:
            init_actions = np.zeros(self.N, dtype=int)
            for i, n in enumerate(self.nums_actions):
                init_actions[i] = np.random.randint(n)
        self.current_actions[:] = init_actions

        for i, player in enumerate(self.players):
            for j in self.opponent_list[i]:
                if j>i:
                    k = j - 1
                player.current_assessment[k][:] = \
                    pure2mixed(self.assessment_sizes[i][k], init_actions[j])

    def play(self):
        """
        The method used to proceed the game by one period. Players take their
        best response strategies given assessments about opponents actions.

        """
        for i, player in enumerate(self.players):
            self.current_actions[i] = \
                player.best_response(player.current_assessment if self.N > 2 else player.current_assessment[0],
                                    tie_breaking=self.tie_breaking)

    def update_asseccements(self, step_size):
        """
        This method used to update each players' assessments. Assessments are updated by
        opponent's action and `step_size`.

        """
        for i, player in enumerate(self.players):
            for j in self.opponent_list[i]:
                if j > i:
                    k = j - 1
                player.current_assessment[k][:] *= 1 - step_size
                player.current_assessment[k][self.current_actions[j]] += step_size

    def _sum_index(self, i, j):
        #i and j are index
        total = 0
        if i==0:
            if j==0:
                total = 0
            else:
                for k in range(j):
                    total += self.assessment_sizes[0][k]
        else:
            for k in range(i):
                total += sum(self.assessment_sizes[k])
            for l in range(j):
                total += self.assessment_sizes[i][l]
        return total

    def get_time_series(self, ts_length, init_actions=None):
        """
        Return the assessments of each players in each round.

        Parameters
        ----------
        ts_length : scalar(int)
            The number of rounds you play in a simulation.

        """
        assessments_sequence = np.empty((ts_length, sum(self.nums_actions)*(self.N-1)))
        assessments_iter = self.get_time_series_iter(ts_length, init_actions)

        for t, assessments in enumerate(assessments_iter):
            for i in range(self.N):
                for j in range(self.N-1):
                    assessments_sequence[t, self._sum_index(i,j):self._sum_index(i,j+1)] = assessments[i][j]

        assessment_list = []
        for i in range(self.N):
            temp = []
            for j in range(self.N-1):
                temp.append(np.asarray(assessments_sequence[:, self._sum_index(i,j):self._sum_index(i,j+1)]))
            assessment_list.append(temp)

        return tuple(assessment_list)

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
            yield self.current_assessments
            self.play()
            self.update_asseccements(self.step_size(t+1))

    def iterate_result(self, ts_length, init_actions=None):
        """
        Returns the ultimate assessments of each players after `ts_length` times play.

        Parameters
        ----------
        ts_length : scalar(int)
            The number of periods you play in a simulation.

        """
        self.set_init_actions(init_actions)

        for t in range(ts_length):
            self.play()
            self.update_asseccements(self.step_size(t+1))
        return self.current_assessments


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
        for j in range(self.N):
            payoff_perturbations.append(random_values[self._sum_index(j,0):self._sum_index(j+1,0)])

        for i, player in enumerate(self.players):
            self.current_actions[i] = player.best_response(
                player.current_assessment if self.N > 2 else player.current_assessment[0],
                tie_breaking=self.tie_breaking,
                payoff_perturbation=payoff_perturbations[i]
            )