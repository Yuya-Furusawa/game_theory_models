"""
Filename: test_fictplay.py
Tests for fictplay.py
"""


import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.stats import norm

from fictplay import FictitiousPlay, StochasticFictitiousPlay


class Test_FictitiousPlay_DecreaingGain:

    def setUp(self):
        '''Setup a FictitiousPlay instance'''
        # symmetric 2x2 coordination game
        matching_pennies = [[( 1, -1), (-1,  1)],
                            [(-1,  1), ( 1, -1)]]
        self.fp = FictitiousPlay(matching_pennies)

    def test_play(self):
        init_actions = (0, 0)
        best_responses = [np.asarray([1,0]), np.asarray([0.5,0.5])]
        assert_array_almost_equal(self.fp.play(init_actions=init_actions), best_responses)

    def test_time_series(self):
        x = self.fp.time_series(ts_length=3, init_actions=(0, 0))
        assert_array_almost_equal(x[0],
            [[1, 0],
             [1, 0],
             [1, 0]])
        assert_array_almost_equal(x[1],
            [[1,0],
             [1/2,1/2],
             [1/3,2/3]])


class Test_FictitiousPlay_ConstantGain:

    def setUp(self):
        matching_pennies = [[( 1, -1), (-1,  1)],
                            [(-1,  1), ( 1, -1)]]
        self.fp = FictitiousPlay(matching_pennies,gain=0.1)

    def test_play(self):
        init_actions = (0, 0)
        best_responses = [np.asarray([1,0]), np.asarray([0.9,0.1])]
        assert_array_almost_equal(self.fp.play(init_actions=init_actions), best_responses)

    def test_time_series(self):
        x = self.fp.time_series(ts_length=3, init_actions=(0, 0))
        assert_array_almost_equal(x[0],
            [[1, 0],
             [1, 0],
             [1, 0]]
            )
        assert_array_almost_equal(x[1],
            [[1,0],
             [0.9,0.1],
             [0.81,0.19]])

class Test_StochasticFictitiosuPlay_DecreaingGain:

    def setUp(self):
        matching_pennies = [[( 1, -1), (-1,  1)],
                            [(-1,  1), ( 1, -1)]]
        distribution = norm()
        self.fp = StochasticFictitiousPlay(matching_pennies,
                                           distribution=distribution)

    def test_play(self):
        x = [np.asarray([1, 0]), np.asarray([0.5, 0.5])]
        assert_array_almost_equal(self.fp.play(init_actions=(0, 0),
                                    random_state=np.random.RandomState(1234)),
                                  x)

    def test_time_series(self):
        x = self.fp.time_series(ts_length=3, init_actions=(0, 0),
                                random_state=np.random.RandomState(1234))
        assert_array_almost_equal(x[0],
            [[1, 0],
             [1, 0],
             [2/3, 1/3]]
            )
        assert_array_almost_equal(x[1],
            [[1, 0],
             [0.5, 0.5],
             [1/3, 2/3]])

class Test_StochasticFictitiosuPlay_ConstantGain:

    def setUp(self):
        matching_pennies = [[( 1, -1), (-1,  1)],
                            [(-1,  1), ( 1, -1)]]
        distribution = norm()
        self.fp = StochasticFictitiousPlay(matching_pennies,
                                           distribution=distribution,
                                           gain=0.1)

    def test_play(self):
        x = [np.asarray([1, 0]), np.asarray([0.9, 0.1])]
        assert_array_almost_equal(self.fp.play(init_actions=(0, 0),
                                    random_state=np.random.RandomState(1234)),
                                  x)

    def test_time_series(self):
        x = self.fp.time_series(ts_length=3, init_actions=(0, 0),
                                random_state=np.random.RandomState(1234))
        assert_array_almost_equal(x[0],
            [[1, 0],
             [1, 0],
             [0.9, 0.1]]
            )
        assert_array_almost_equal(x[1],
            [[1, 0],
             [0.9, 0.1],
             [0.81, 0.19]])

# Invalid inputs #

if __name__ == '__main__':
    import sys
    import nose

    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)