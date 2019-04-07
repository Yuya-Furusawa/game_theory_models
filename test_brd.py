"""
Filename: test_brd.py

Tests for brd.py

"""
import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import eq_, ok_

from brd import BRD, KMR, SamplingBRD


class TestBRD:
    '''Test the methods of BRD'''

    def setUp(self):
        '''Setup a BRD instance'''
        # 2x2 coordination game with action 1 risk-dominant
        payoff_matrix = [[4, 0],
                         [3, 2]]
        self.N = 4  # 4 players
        self.brd = BRD(payoff_matrix, self.N)

    def test_time_series_1(self):
        assert_array_equal(
            self.brd.time_series(ts_length=3, init_actions=(0,0,0,0)),
            [[4, 0],
             [4, 0],
             [4, 0]]
            )

    def test_time_series_2(self):
        np.random.seed(22)
        assert_array_equal(
            self.brd.time_series(ts_length=3, init_actions=(0,0,1,1)),
            [[2, 2],
             [1, 3],
             [0, 4]]
            )

class TestKMR:
    '''Test the methods of KMR'''

    def setUp(self):
        payoff_matrix = [[4, 0],
                         [3, 2]]
        self.N = 4
        self.kmr = KMR(payoff_matrix, self.N)

    def test_time_series(self):
        series = self.kmr.time_series(ts_length=5, init_actions=(0,0,0,0))
        for t in range(5):
            eq_(sum(series[t, :]), 4)

class TestSamplingBRD:
    '''Test the methods of SamplingBRD'''

    def setUp(self):
        payoff_matrix = [[4, 0],
                         [3, 2]]
        self.N = 4
        self.sbrd = SamplingBRD(payoff_matrix, self.N)

    def test_time_series(self):
        series = self.sbrd.time_series(ts_length=5, init_actions=(0,0,0,0))
        for t in range(5):
            eq_(sum(series[t, :]), 4)



if __name__ == '__main__':
    import sys
    import nose

    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
