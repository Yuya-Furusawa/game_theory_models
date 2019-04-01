import numpy as np
from util import check_random_state
from normal_form_game import *
from random_game import random_pure_actions


class FictitiousPlay():
    """
    Class representing the fictitious play model.

    Parameters
    ----------
    data : NormalFormGame or array-like
        The game played in fictitious play model

    gain : scalar(float), optional(default=None)
        The gain of fictitous play model. If gain is None, the model becomes
        decreasing gain model. If gain is scalar, the model becomes constant
        constant gain model.

    Attributes
    ----------
    g : NomalFormGame
        The game. See Parameters.

    N : scalar(int)
        The number of players in the model.

    players : Player
        Player instance in the model.

    nums_actions : tuple(int)
        Tuple of the number of actions, one for each player.
    """

    def __init__(self, data, gain=None):
        if isinstance(data, NormalFormGame):
            self.g = data
        else:  # data must be array_like
            payoffs = np.asarray(data)
            self.g = NormalFormGame(payoffs)

        self.N = self.g.N
        self.players = self.g.players
        self.nums_actions = self.g.nums_actions
        self.tie_breaking = 'smallest'

        if gain is None:
            self.step_size = lambda t: 1 / (t+1)  # decreasing gain
        else:
            self.step_size = lambda t: gain  # constant gain

    def _play(self, actions, t):
        brs = np.zeros(self.N, dtype=int)
        for i, player in enumerate(self.players):
            index = [j for j in range(i+1, self.N)]
            index.extend([j for j in range(i)])
            opponent_actions = np.asarray([actions[i] for i in index])
            brs[i] = player.best_response(
                opponent_actions if self.N > 2 else opponent_actions[0],
                tie_breaking=self.tie_breaking)

        for i in range(self.N):
            actions[i][:] *= 1 - self.step_size(t+1)
            actions[i][brs[i]] += self.step_size(t+1)

        return actions

    def play(self, init_actions=None, num_reps=1, t_init=0, random_state=None):
        """
        Return a new action profile which is updated by playing the game
        `num_reps` times.

        Parameters
        ----------
        init_actions : tuple(int), optional(default=None)
            The action profile in the first period. If None, selected randomly.

        num_reps : scalar(int), optional(default=1)
            The number of iterations.

        t_init : scalar(int), optional(default=0)
            The period the game starts.

        Returns
        -------
        tuple(int)
            The action profile after iteration.
        """
        if init_actions is None:
            init_actions = random_pure_actions(self.nums_actions, random_state)
        actions = [i for i in init_actions]
        for i in range(self.N):
            actions[i] = pure2mixed(self.nums_actions[i], init_actions[i])
        for t in range(num_reps):
            actions = self._play(actions, t+t_init)
        return actions

    def time_series(self, ts_length, init_actions=None, t_init=0,
                    random_state=None):
        """
        Return the array representing time series of normalized action history.

        Parameters
        ----------
        ts_length : scalar(int)
            The number of iterations.

        init_actions : tuple(int), optional(default=None)
            The action profile in the first period. If None, selected randomly.

        t_init : scalar(int), optional(default=0)
            The period the game starts.

        Returns
        -------
        Array
            The array representing time series of normalized action history.
        """
        if init_actions is None:
            init_actions = random_pure_actions(self.nums_actions, random_state)
        out = [np.empty((ts_length, self.nums_actions[i]))
               for i in range(self.N)]
        actions = [np.empty(self.nums_actions[i]) for i in range(self.N)]
        for i in range(self.N):
            actions[i] = pure2mixed(self.nums_actions[i], init_actions[i])[:]
        for t in range(ts_length):
            for i in range(self.N):
                out[i][t, :] = actions[i][:]
            actions = self._play(actions, t+t_init)
        return out


class StochasticFictitiousPlay(FictitiousPlay):
    """
    Class representing the stoochastic fictitous play model.
    Subclass of FictitiousPlay.

    Parameters
    ----------
    data : NormalFormGame or array-like
        The game played in the stochastic fictitious play model.

    gain : scalar(int), optional(default=None)
        The gain of fictitous play model. If gain is None, the model becomes
        decreasing gain model. If gain is scalar, the model becomes constant
        constant gain model.

    distribution : {'extreme', 'normal'}, optional(default='extreme')
        The distribution of payoff shocks. If 'extreme', the distribution is
        type I extreme distribution. If 'normal', the distribution is standard
        normal distribution.

    Attributes
    ----------
    See attributes of FictitousPlay.
    """

    def __init__(self, data, gain=None,
                 distribution='extreme', random_state=None):
        FictitiousPlay.__init__(self, data, gain)

        random_state = check_random_state(random_state)
        if distribution == 'extreme':
            loc = -np.euler_gamma * np.sqrt(6) / np.pi
            scale = np.sqrt(6) / np.pi
            self.payoff_perturbation_dist = \
                lambda size: random_state.gumbel(
                                                 loc=loc, scale=scale, size=size
                                                )
        elif distribution == 'normal':  # standard normal distribution
            self.payoff_perturbation_dist = \
                lambda size: random_state.standard_normal(size=size)
        else:
            raise ValueError("`distribution` must be 'extreme' or 'normal'")

        self.tie_breaking = 'smallest'

    def _play(self, actions, t):
        brs = np.zeros(self.N, dtype=int)
        for i, player in enumerate(self.players):
            index = [j for j in range(i+1, self.N)]
            index.extend([j for j in range(i)])
            opponent_actions = np.asarray([actions[i] for i in index])
            payoff_perturbation = \
                self.payoff_perturbation_dist(size=self.nums_actions[i])
            brs[i] = player.best_response(
                opponent_actions if self.N > 2 else opponent_actions[0],
                tie_breaking=self.tie_breaking,
                payoff_perturbation=payoff_perturbation)

        for i in range(self.N):
            actions[i][:] *= 1 - self.step_size(t+1)
            actions[i][brs[i]] += self.step_size(t+1)

        return actions
