r"""
Filename: normal_form_game.py

Authors: Tomohiro Kusano, Daisuke Oyama

Tools for normal form games.

Definitions and Basic Concepts
------------------------------

An :math:`N`-player *normal form game* :math:`g = (I, (A_i)_{i \in I},
(u_i)_{i \in I})` consists of

- the set of *players* :math:`I = \{0, \ldots, N-1\}`,
- the set of *actions* :math:`A_i = \{0, \ldots, n_i-1\}` for each
  player :math:`i \in I`, and
- the *payoff function* :math:`u_i \colon A_i \times A_{i+1} \times
  \cdots \times A_{i+N-1} \to \mathbb{R}` for each player :math:`i \in
  I`,

where :math:`i+j` is understood modulo :math:`N`. Note that we adopt the
convention that the 0-th argument of the payoff function :math:`u_i` is
player :math:`i`'s own action and the :math:`j`-th argument is player
(:math:`i+j`)'s action (modulo :math:`N`). A mixed action for player
:math:`i` is a probability distribution on :math:`A_i` (while an element
of :math:`A_i` is referred to as a pure action). A pure action
:math:`a_i \in A_i` is identified with the mixed action that assigns
probability one to :math:`a_i`. Denote the set of mixed actions of
player :math:`i` by :math:`X_i`. We also denote :math:`A_{-i} = A_{i+1}
\times \cdots \times A_{i+N-1}` and :math:`X_{-i} = X_{i+1} \times
\cdots \times X_{i+N-1}`.

The (pure-action) *best response correspondence* :math:`b_i \colon
X_{-i} \to A_i` for each player :math:`i` is defined by

.. math::

    b_i(x_{-i}) = \{a_i \in A_i \mid
        u_i(a_i, x_{-i}) \geq u_i(a_i', x_{-i})
        \ \forall\,a_i' \in A_i\},

where :math:`u_i(a_i, x_{-i}) = \sum_{a_{-i} \in A_{-i}} u_i(a_i,
a_{-i}) \prod_{j=1}^{N-1} x_{i+j}(a_j)` is the expected payoff to action
:math:`a_i` against mixed actions :math:`x_{-i}`. A profile of mixed
actions :math:`x^* \in X_0 \times \cdots \times X_{N-1}` is a *Nash
equilibrium* if for all :math:`i \in I` and :math:`a_i \in A_i`,

.. math::

    x_i^*(a_i) > 0 \Rightarrow a_i \in b_i(x_{-i}^*),

or equivalently, :math:`x_i^* \cdot v_i(x_{-i}^*) \geq x_i \cdot
v_i(x_{-i}^*)` for all :math:`x_i \in X_i`, where :math:`v_i(x_{-i})` is
the vector of player :math:`i`'s payoffs when the opponent players play
mixed actions :math:`x_{-i}`.

Creating a NormalFormGame
-------------------------

There are three ways to construct a `NormalFormGame` instance.

The first is to pass an array of payoffs for all the players:

>>> matching_pennies_bimatrix = [[(1, -1), (-1, 1)], [(-1, 1), (1, -1)]]
>>> g = NormalFormGame(matching_pennies_bimatrix)
>>> g.players[0]
Player([[ 1, -1],
        [-1,  1]])
>>> g.players[1]
Player([[-1,  1],
        [ 1, -1]])

If a square matrix (2-dimensional array) is given, then it is considered
to be a symmetric two-player game:

>>> coordination_game_matrix = [[4, 0], [3, 2]]
>>> g = NormalFormGame(coordination_game_matrix)
>>> g
NormalFormGame([[[4, 4],  [0, 3]],
                [[3, 0],  [2, 2]]])

The second is to specify the sizes of the action sets of the players,
which gives a `NormalFormGame` instance filled with payoff zeros, and
then set the payoff values to each entry:

>>> g = NormalFormGame((2, 2))
>>> g
NormalFormGame([[[ 0.,  0.],  [ 0.,  0.]],
                [[ 0.,  0.],  [ 0.,  0.]]])
>>> g[0, 0] = 1, 1
>>> g[0, 1] = -2, 3
>>> g[1, 0] = 3, -2
>>> g
NormalFormGame([[[ 1.,  1.],  [-2.,  3.]],
                [[ 3., -2.],  [ 0.,  0.]]])

The third is to pass an array of `Player` instances, as explained in the
next section.

Creating a Player
-----------------

A `Player` instance is created by passing a payoff array:

>>> from normal_form_game import Player
>>> player0 = Player([[3, 1], [0, 2]])
>>> player0
Player([[3, 1],
        [0, 2]])

Passing an array of `Player` instances is the third way to create a
`NormalFormGame` instance.

>>> player1 = Player([[2, 0], [1, 3]])
>>> g = NormalFormGame((player0, player1))
>>> g
NormalFormGame([[[3, 2],  [1, 1]],
                [[0, 0],  [2, 3]]])

Beware that in `payoff_array[h, k]`, `h` refers to the player's own
action, while `k` refers to the opponent player's action.

"""
import re
import numbers
import numpy as np
from util import check_random_state


class Player(object):
    """
    Class representing a player in an N-player normal form game.

    Parameters
    ----------
    payoff_array : array_like(float)
        Array representing the player's payoff function, where
        payoff_array[a_0, a_1, ..., a_{N-1}] is the payoff to the player
        when the player plays action a_0 while his N-1 opponents play
        actions a_1, ..., a_{N-1}, respectively.

    Attributes
    ----------
    payoff_array : ndarray(float, ndim=N)
        See Parameters.

    num_actions : scalar(int)
        The number of actions available to the player.

    num_opponents : scalar(int)
        The number of opponent players.

    """
    def __init__(self, payoff_array):
        self.payoff_array = np.asarray(payoff_array)

        if self.payoff_array.ndim == 0:
            raise ValueError('payoff_array must be an array_like')

        self.num_opponents = self.payoff_array.ndim - 1
        self.num_actions = self.payoff_array.shape[0]

        self.tol = 1e-8

    def __repr__(self):
        s = _payoff_array2string(self.payoff_array,
                                 class_name=self.__class__.__name__)
        return s

    def __str__(self):
        N = self.num_opponents + 1
        s = 'Player in a {N}-player normal form game'.format(N=N)
        s += ' with payoff array:\n'
        s += np.array2string(self.payoff_array, separator=', ')
        return s

    def payoff_vector(self, opponents_actions):
        """
        Return an array of payoff values, one for each own action, given
        a profile of the opponents' actions.

        Parameters
        ----------
        opponents_actions : see `best_response`.

        Returns
        -------
        payoff_vector : ndarray(float, ndim=1)
            An array representing the player's payoff vector given the
            profile of the opponents' actions.

        """
        def reduce_last_player(payoff_array, action):
            """
            Given `payoff_array` with ndim=M, return the payoff array
            with ndim=M-1 fixing the last player's action to be `action`.

            """
            if isinstance(action, numbers.Integral):  # pure action
                return payoff_array.take(action, axis=-1)
            else:  # mixed action
                return payoff_array.dot(action)

        if self.num_opponents == 1:
            payoff_vector = \
                reduce_last_player(self.payoff_array, opponents_actions)
        elif self.num_opponents >= 2:
            payoff_vector = self.payoff_array
            for i in reversed(range(self.num_opponents)):
                payoff_vector = \
                    reduce_last_player(payoff_vector, opponents_actions[i])
        else:  # Trivial case with self.num_opponents == 0
            payoff_vector = self.payoff_array

        return payoff_vector

    def is_best_response(self, own_action, opponents_actions):
        """
        Return True if `own_action` is a best response to
        `opponents_actions`.

        Parameters
        ----------
        own_action : scalar(int) or array_like(float, ndim=1)
            An integer representing a pure action, or an array of floats
            representing a mixed action.

        opponents_actions : see `best_response`

        Returns
        -------
        bool
            True if `own_action` is a best response to
            `opponents_actions`; False otherwise.

        """
        payoff_vector = self.payoff_vector(opponents_actions)
        payoff_max = payoff_vector.max()

        if isinstance(own_action, numbers.Integral):
            return payoff_vector[own_action] >= payoff_max - self.tol
        else:
            return np.dot(own_action, payoff_vector) >= payoff_max - self.tol

    def best_response(self, opponents_actions, tie_breaking='smallest',
                      payoff_perturbation=None, random_state=None):
        """
        Return the best response action(s) to `opponents_actions`.

        Parameters
        ----------
        opponents_actions : array_like(int or array_like(float)) or
                            array_like(int, ndim=1) or scalar(int)
            A profile of N-1 opponents' actions. If N=2, then it must be
            a 1-dimensional array of floats (in which case it is treated
            as the opponent's mixed action) or a scalar of integer (in
            which case it is treated as the opponent's pure action). If
            N>2, then it must be an array of N-1 objects, where each
            object must be an integer (pure action) or an array of
            floats (mixed action).

        tie_breaking : {'smallest', 'random', False},
                       optional(default='smallest')
            Control how, or whether, to break a tie (see Returns for
            details).

        payoff_perturbation : array_like(float), optional(default=None)
            Array of length equal to the number of actions of the player
            containing the values ("noises") to be added to the payoffs
            in determining the best response.

        random_state : scalar(int) or np.random.RandomState,
                       optional(default=None)
            Random seed (integer) or np.random.RandomState instance to
            set the initial state of the random number generator for
            reproducibility. If None, a randomly initialized RandomState
            is used. Relevant only when tie_breaking='random'.

        Returns
        -------
        scalar(int) or ndarray(int, ndim=1)
            If tie_breaking=False, returns an array containing all the
            best response pure actions. If tie_breaking='smallest',
            returns the best response action with the smallest index; if
            tie_breaking='random', returns an action randomly chosen
            from the best response actions.

        """
        payoff_vector = self.payoff_vector(opponents_actions)
        if payoff_perturbation is not None:
            payoff_vector += payoff_perturbation

        if tie_breaking == 'smallest':
            best_response = np.argmax(payoff_vector)
            return best_response
        else:
            best_responses = \
                np.where(payoff_vector >= payoff_vector.max() - self.tol)[0]
            if tie_breaking == 'random':
                return self.random_choice(best_responses,
                                          random_state=random_state)
            elif tie_breaking is False:
                return best_responses
            else:
                msg = "tie_breaking must be one of 'smallest', 'random' " + \
                      "or False"
                raise ValueError(msg)

    def random_choice(self, actions=None, random_state=None):
        """
        Return a pure action chosen randomly from `actions`.

        Parameters
        ----------
        actions : array_like(int), optional(default=None)
            An array of integers representing pure actions.

        random_state : scalar(int) or np.random.RandomState,
                       optional(default=None)
            Random seed (integer) or np.random.RandomState instance to
            set the initial state of the random number generator for
            reproducibility. If None, a randomly initialized RandomState
            is used.

        Returns
        -------
        scalar(int)
            If `actions` is given, returns an integer representing a
            pure action chosen randomly from `actions`; if not, an
            action is chosen randomly from the player's all actions.

        """
        random_state = check_random_state(random_state)

        if actions is not None:
            n = len(actions)
        else:
            n = self.num_actions

        if n == 1:
            idx = 0
        else:
            idx = random_state.randint(n)

        if actions is not None:
            return actions[idx]
        else:
            return idx


class NormalFormGame(object):
    """
    Class representing an N-player normal form game.

    Parameters
    ----------
    data : array_like(Player) or array_like(int, ndim=1) or
           array_like(float, ndim=2 or N+1)
        Data to initialize a NormalFormGame. `data` may be an array of
        Players, in which case the shapes of the Players' payoff arrays
        must be consistent. If `data` is an array of N integers, then
        these integers are treated as the numbers of actions of the N
        players and a NormalFormGame is created consisting of payoffs
        all 0 with `data[i]` actions for each player `i`. `data` may
        also be an (N+1)-dimensional array representing payoff profiles.
        If `data` is a square matrix (2-dimensional array), then the
        game will be a symmetric two-player game where the payoff matrix
        of each player is given by the input matrix.

    Attributes
    ----------
    players : tuple(Player)
        Tuple of the Player instances of the game.

    N : scalar(int)
        The number of players.

    nums_actions : tuple(int)
        Tuple of the numbers of actions, one for each player.

    """
    def __init__(self, data):
        # data represents an array_like of Players
        if hasattr(data, '__getitem__') and isinstance(data[0], Player):
            N = len(data)

            # Check that the shapes of the payoff arrays are consistent
            shape_0 = data[0].payoff_array.shape
            for i in range(1, N):
                shape = data[i].payoff_array.shape
                if not (
                    len(shape) == N and
                    shape == shape_0[i:] + shape_0[:i]
                ):
                    raise ValueError(
                        'shapes of payoff arrays must be consistent'
                    )

            self.players = tuple(data)

        # data represents action sizes or a payoff array
        else:
            data = np.asarray(data)

            if data.ndim == 0:  # data represents action size
                # Trivial game consisting of one player
                N = 1
                self.players = (Player(np.zeros(data)),)

            elif data.ndim == 1:  # data represents action sizes
                N = data.size
                # N instances of Player created
                # with payoff_arrays filled with zeros
                # Payoff values set via __setitem__
                self.players = tuple(
                    Player(np.zeros(tuple(data[i:]) + tuple(data[:i])))
                    for i in range(N)
                )

            elif data.ndim == 2 and data.shape[1] >= 2:
                # data represents a payoff array for symmetric two-player game
                # Number of actions must be >= 2
                if data.shape[0] != data.shape[1]:
                    raise ValueError(
                        'symmetric two-player game must be represented ' +
                        'by a square matrix'
                    )
                N = 2
                self.players = tuple(Player(data) for i in range(N))

            else:  # data represents a payoff array
                # data must be of shape (n_0, ..., n_{N-1}, N),
                # where n_i is the number of actions available to player i,
                # and the last axis contains the payoff profile
                N = data.ndim - 1
                if data.shape[-1] != N:
                    raise ValueError(
                        'size of innermost array must be equal to ' +
                        'the number of players'
                    )
                self.players = tuple(
                    Player(
                        data.take(i, axis=-1).transpose(list(range(i, N)) +
                                                        list(range(i)))
                    ) for i in range(N)
                )

        self.N = N  # Number of players
        self.nums_actions = tuple(
            player.num_actions for player in self.players
        )

    @property
    def payoff_profile_array(self):
        N = self.N
        # To infer the dype
        dtype = np.dtype(np.sum([player.payoff_array.take(0)
                                 for player in self.players]))
        payoff_profile_array = \
            np.empty(self.players[0].payoff_array.shape + (N,), dtype=dtype)
        for i, player in enumerate(self.players):
            payoff_profile_array[..., i] = \
                player.payoff_array.transpose(list(range(N-i, N)) +
                                              list(range(N-i)))
        return payoff_profile_array

    def __repr__(self):
        s = _payoff_profile_array2string(self.payoff_profile_array,
                                         class_name=self.__class__.__name__)
        return s

    def __str__(self):
        s = '{N}-player NormalFormGame'.format(N=self.N)
        s += ' with payoff profile array:\n'
        s += _payoff_profile_array2string(self.payoff_profile_array)
        return s

    def __getitem__(self, action_profile):
        if self.N == 1:  # Trivial game with 1 player
            if not isinstance(action_profile, numbers.Integral):
                raise TypeError('index must be an integer')
            return self.players[0].payoff_array[action_profile]

        # Non-trivial game with 2 or more players
        try:
            if len(action_profile) != self.N:
                raise IndexError('index must be of length {0}'.format(self.N))
        except TypeError:
            raise TypeError('index must be a tuple')

        payoff_profile = [
            player.payoff_array[
                tuple(action_profile[i:]) + tuple(action_profile[:i])
            ]
            for i, player in enumerate(self.players)
        ]

        return payoff_profile

    def __setitem__(self, action_profile, payoff_profile):
        if self.N == 1:  # Trivial game with 1 player
            if not isinstance(action_profile, numbers.Integral):
                raise TypeError('index must be an integer')
            self.players[0].payoff_array[action_profile] = payoff_profile
            return None

        # Non-trivial game with 2 or more players
        try:
            if len(action_profile) != self.N:
                raise IndexError('index must be of length {0}'.format(self.N))
        except TypeError:
            raise TypeError('index must be a tuple')

        try:
            if len(payoff_profile) != self.N:
                raise ValueError(
                    'value must be an array_like of length {0}'.format(self.N)
                )
        except TypeError:
            raise TypeError('value must be a tuple')

        for i, player in enumerate(self.players):
            player.payoff_array[
                tuple(action_profile[i:]) + tuple(action_profile[:i])
            ] = payoff_profile[i]

    def is_nash(self, action_profile):
        """
        Return True if `action_profile` is a Nash equilibrium.

        Parameters
        ----------
        action_profile : array_like(int or array_like(float))
            An array of N objects, where each object must be an integer
            (pure action) or an array of floats (mixed action).

        Returns
        -------
        bool
            True if `action_profile` is a Nash equilibrium; False
            otherwise.

        """
        if self.N == 2:
            for i, player in enumerate(self.players):
                own_action, opponent_action = \
                    action_profile[i], action_profile[1-i]
                if not player.is_best_response(own_action, opponent_action):
                    return False

        elif self.N >= 3:
            for i, player in enumerate(self.players):
                own_action = action_profile[i]
                opponents_actions = \
                    tuple(action_profile[i+1:]) + tuple(action_profile[:i])

                if not player.is_best_response(own_action, opponents_actions):
                    return False

        else:  # Trivial case with self.N == 1
            if not self.players[0].is_best_response(action_profile[0], None):
                return False

        return True


def _payoff_array2string(payoff_array, class_name=None):
    prefix, suffix = '', ''
    if class_name is not None:
        prefix = class_name + '('
        suffix = ')'
    s = np.array2string(payoff_array, separator=', ', prefix=prefix)
    return prefix + s + suffix


def _payoff_profile_array2string(payoff_profile_array, class_name=None):
    s = np.array2string(payoff_profile_array, separator=', ')

    # Remove one linebreak
    s = re.sub(r'(\n+)', lambda x: x.group(0)[0:-1], s)

    if class_name is not None:
        prefix = class_name + '('
        next_line_prefix = ' ' * len(prefix)
        suffix = ')'
        l = s.splitlines()
        l[0] = prefix + l[0]
        for i in range(1, len(l)):
            if l[i]:
                l[i] = next_line_prefix + l[i]
        l[-1] += suffix
        s = '\n'.join(l)

    return s


def pure2mixed(num_actions, action):
    """
    Convert a pure action to the corresponding mixed action.

    Parameters
    ----------
    num_actions : scalar(int)
        The number of the pure actions (= the length of a mixed action).

    action : scalar(int)
        The pure action to convert to the corresponding mixed action.

    Returns
    -------
    ndarray(float, ndim=1)
        The mixed action representation of the given pure action.

    """
    mixed_action = np.zeros(num_actions)
    mixed_action[action] = 1
    return mixed_action
