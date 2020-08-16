import unittest
from Functional_FBSNN_JAXvecOptimLaxScanZ import fetch_minibatch

class Tests(unittest.TestCase):
    def setUp(self):
        pass

    def test_fetch_minibatch(self):
        M = 98  # number of trajectories (batch size)
        N = 50  # number of time snapshots
        D = 100  # 50#100  # number of dimensions
        T = 1.0
        result = fetch_minibatch(T,M,N,D)
        self.assertTrue(result.shape == (M,N,D))
