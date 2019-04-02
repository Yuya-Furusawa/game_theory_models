"""
Filename: test_localint.py
Author: Tomohiro Kusano

Tests for localint.py

"""
from __future__ import division

import numpy as np
from numpy.testing import assert_array_equal, assert_equal
from nose.tools import eq_, ok_, raises

from localint import LocalInteraction


class TestLocalInteraction:
    '''Test the methods of LocalInteraction'''

    def setUp(self):
        '''Setup a LocalInteraction instance'''
        payoff_matrix = np.asarray([[4,0],[2,3]])
        adj_matrix = np.asarray([[0, 1, 3],
                                 [2, 0, 1],
                                 [3, 2, 0]])
        self.li = LocalInteraction(payoff_matrix, adj_matrix)

    def test_play(self):
        init_actions = (0,0,1)
        x = (1,0,0)
        assert_equal(self.li.play(init_actions=init_actions), x)

    def test_time_series(self):
        init_actions = (0,0,1)
        x = [[0,0,1],
             [1,0,0],
             [0,1,1]]
        assert_array_equal(self.li.time_series(ts_length=3,
                                               init_actions=init_actions), x)


# Invalid inputs #

@raises(ValueError)
def test_localint_invalid_input_nonsquare_adj_matrix():
    li = LocalInteraction(payoff_matrix=np.zeros((2, 2)),
                          adj_matrix=np.zeros((2, 3)))


@raises(ValueError)
def test_localint_invalid_input_nonsquare_payoff_matrix():
    li = LocalInteraction(payoff_matrix=np.zeros((2, 3)),
                          adj_matrix=np.zeros((2, 2)))


if __name__ == '__main__':
    import sys
    import nose

    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
