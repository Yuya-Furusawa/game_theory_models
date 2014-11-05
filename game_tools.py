"""
Filename: game_tools.py

Authors: Tomohiro Kusano, Daisuke Oyama

Tools for Game Theory.

"""
from __future__ import division

import numpy as np


class Player_2P(object):
    """
    Class representing a player in a two-player normal form game.

    TODO: Extend it to the general case of N-1 opponent players

    Parameters
    ----------
    payoff_matrix : array_like(float)
        A 2-dimensional ndarray representing a payoff matrix.

    init_action : int?, optional(default=None)
        (To be written)

    Attributes
    ----------
    payoff_matrix : ndarray(float)
        The payoff matrix of the game converted into ndarray.

    n : long
        The number of rows of payoff_matrix.

    m : long
        The number of columns of payoff_matrix.

    current_action : int?
        (To be written)

    Examples
    --------
    >>> P = game_tools.Player_2P([[4,0], [3,2]])
    >>> P
    Player_2P:
        payoff_matrix: array([[4,0],[3,2]])
        n: 2L
        m: 2L

    """
    def __init__(self, payoff_matrix, init_action=None):
        self.payoff_matrix = np.asarray(payoff_matrix)

        if len(self.payoff_matrix.shape) != 2:  # ndim?
            raise ValueError('payoff matrix must be 2-dimensional')

        self.n, self.m = self.payoff_matrix.shape

        self.current_action = init_action

    def best_response(self, opponent_action,
                      tie_breaking=True, payoff_perturbations=None):
        """
        Return the best response actions to the given opponent action.

        Parameters
        ----------

        Returns
        -------

        """
        br_actions = br_correspondence(opponent_action, self.payoff_matrix)

        if tie_breaking:
            return random_choice(br_actions)
        else:
            return br_actions

    def random_choice(self):
        """
        Return a pure action chosen at random from the player's actions.

        Parameters
        ----------

        Returns
        -------

        """
        return random_choice(range(self.n))


class NormalFormGame_2P(object):
    """
    Class representing a two-player normal form game.

    Parameters
    ----------

    Attributes
    ----------

    Examples
    --------

    """
    player_indices = [0, 1]

    def __init__(self, payoffs):
        payoffs_ndarray = np.asarray(payoffs)
        if payoffs_ndarray.ndim == 3:  # bimatrix game
            self.bimatrix = payoffs_ndarray
            self.matrices = [payoffs_ndarray[:, :, 0],
                             payoffs_ndarray[:, :, 1].T]
        elif payoffs_ndarray.ndim == 2:  # symmetric game
            try:
                self.bimatrix = np.dstack((payoffs_ndarray, payoffs_ndarray.T))
            except:
                raise ValueError('a symmetric game must be a square array')
            self.matrices = [payoffs_ndarray, payoffs_ndarray]
        else:
            raise ValueError(
                'the game must be represented by a bimatrix or a square matrix'
            )

        self.players = [
            Player_2P(self.matrices[i]) for i in self.player_indices
        ]


def br_correspondence(opponents_actions, payoffs):
    """
    Best response correspondence in pure actions. It returns a list of
    the pure best responses to a profile of N-1 opponents' actions.

    TODO: Extend it to the general case of N-1 opponent players

    Parameters
    ----------
    opponents_actions : array_like(float, ndim=1 or 2) or
                        array_like(int, ndim=1) or scalar(int)
        A profile of N-1 opponents' actions. If N=2, then it must be a
        1-dimensional array_like of floats (in which case it is treated
        as the opponent's mixed action) or a scalar of integer (in which
        case it is treated as the opponent's pure action). If N>2, then
        it must be a 2-dimensional array_like of floats (profile of
        mixed actions) or a 1-dimensional array_like of integers
        (profile of pure actions).

    payoffs : array_like(float, ndim=N)
        An N-dimensional array_like representing the player's payoffs.

    Returns
    -------
    ndarray(int)
        An array of the pure action best responses to opponents_actions.

    """
    payoffs_ndarray = np.asarray(payoffs)

    if isinstance(opponents_actions, int):
        payoff_vec = payoffs_ndarray[:, opponents_actions]
    else:
        payoff_vec = np.dot(payoffs_ndarray, opponents_actions)
    return np.where(payoff_vec == payoff_vec.max())[0]


def random_choice(actions):
    """
    Choose an action randomly from given actions.

    Parameters
    ----------
    actions : list(int)
        A list of pure actions represented by nonnegative integers.

    Returns
    -------
    int
        A pure action chosen at random from given actions.

    """
    if len(actions) == 1:
        return actions[0]
    else:
        return np.random.choice(actions)


def pure2mixed(num_actions, action):
    """
    Convert a pure action to the corresponding mixed action.

    Parameters
    ----------
    num_actions : int
        The number of pure actions.

    action : int
        The pure action you want to convert to the corresponding
        mixed action.

    Returns
    -------
    ndarray(int) <- float?
        The corresponding mixed action for the given pure action.

    """
    mixed_action = np.zeros(num_actions)
    mixed_action[action] = 1
    return mixed_action
