"""Unit tests for the Lyapunov functions."""

from __future__ import division, print_function, absolute_import

from numpy.testing import assert_allclose, assert_equal
import pytest
import unittest
import numpy as np
import tensorflow as tf
import sys

from safe_learning.functions import (DeterministicFunction, GridWorld)
from safe_learning.lyapunov import (line_search_bisection, Lyapunov,
                                    smallest_boundary_value)

if sys.version_info.major <= 2:
    import mock
else:
    from unittest import mock


class LineSearchTest(unittest.TestCase):
    """Test the line search."""

    def setUp(self):
        """Set up."""
        self.objective = lambda x: x < 0.5

    def test_simple(self):
        """Test a simple binary optimization criterion."""
        atol = 1e-5
        x = line_search_bisection(self.objective, [0, 1], atol)
        assert_allclose(x[0], 0.5, rtol=0, atol=atol)

    def test_lower(self):
        """Test what happens if the constraint cannot be satisfied."""
        x = line_search_bisection(self.objective, [1, 2], 1e-5)
        assert(x is None)

    def test_upper(self):
        """Test what happens if the constraint is trivially satisfied."""
        x = line_search_bisection(self.objective, [0, 0.4], 1e-5)
        assert_equal(x[0], x[1])


class LyapunovTest(unittest.TestCase):
    """Test the Lyapunov base class."""

    def setUp(self):
        """Initialize a lyapunov function."""
        self.discretization = np.array([[0], [1], [2], [3]])
        self.lyapunov_function = lambda x: np.abs(x)
        self.dynamics = lambda x, u: np.zeros_like(x)
        self.epsilon = 1
        self.lyap = Lyapunov(self.discretization, self.lyapunov_function,
                             self.dynamics, self.epsilon)

    def test_errors(self):
        """Test the NotImplementedErrors."""
        pytest.raises(NotImplementedError,
                      self.lyap.v_decrease_confidence, None, None)
        pytest.raises(NotImplementedError, lambda: self.lyap.threshold)

    def test_safe_set_init(self):
        """Test the safe set initialization."""
        initial_set = [0, 1, 0, 1]
        lyap = Lyapunov(self.discretization, self.lyapunov_function,
                        self.dynamics, self.epsilon, initial_set=initial_set)

        initial_set = np.array([False, True, False, True])
        assert_equal(initial_set, lyap.initial_safe_set)
        assert_equal(initial_set, lyap.safe_set)

    @mock.patch('safe_learning.lyapunov.line_search_bisection')
    def test_max_levelset(self, lsb):
        """Test the function to compute the maximum levelset."""
        accuracy = 0.1
        interval = [0, -0.3]
        lsb.return_value = [0., 0.1]
        self.lyap.max_safe_levelset(0.1, interval)
        lsb.assert_called_with(self.lyap._levelset_is_safe,
                               interval, accuracy)

        v = self.lyapunov_function(self.discretization)
        self.lyap.max_safe_levelset(accuracy)

        assert(lsb.call_args[0][1][0] == 0)
        assert_allclose(lsb.call_args[0][1][1], np.max(v) + accuracy)

    def test_levelset_is_safe(self):
        """Test the helper method for safe levelset construction."""
        self.lyap.v_dot_negative = np.array([True, False, False, False])

        assert(self.lyap._levelset_is_safe(0.5))
        assert(not self.lyap._levelset_is_safe(1.1))

        s = self.lyap.max_safe_levelset(0.01)
        assert(s < 1.)
        assert(s >= np.max(self.lyap.V[self.lyap.v_dot_negative]))

    @mock.patch('safe_learning.lyapunov.Lyapunov.threshold',
                new_callable=mock.PropertyMock)
    @mock.patch('safe_learning.lyapunov.Lyapunov.v_decrease_confidence')
    def test_update(self, decrease_confidence, threshold):
        """Test the update step."""
        acc = 0.1
        threshold.return_value = -0.15
        decrease_confidence.return_value = np.array([-0.5, -0.2, 0, -1]), 0

        self.lyap.update_safe_set(acc)

        assert(self.lyap.cmax < 2)
        assert(self.lyap.cmax >= np.max(self.lyap.V[:2]))

        assert_equal(self.lyap.safe_set, np.array([True, True, False, False]))
        assert_equal(self.lyap.v_dot_negative,
                     np.array([True, True, False, True]))

        self.lyap.initial_safe_set = np.array([False, False, True, False])
        self.lyap.update_safe_set(acc)
        assert(self.lyap.cmax >= 3)
        assert(self.lyap.cmax <= 3 + acc)

        assert(np.all(self.lyap.safe_set))
        assert(np.all(self.lyap.v_dot_negative))

        # Test uncertain dynamics.
        dynamics = lambda x, u: (np.array([3.2]), np.array([1.4]))

        v1 = np.array([-0.5, -0.5, -0.5, -0.5])
        v2 = np.array([0., 0.4, -0.3, 0.6])
        decrease_confidence.return_value = (v1, v2)
        lyap = Lyapunov(self.discretization, self.lyapunov_function,
                        dynamics, self.epsilon)
        lyap.update_safe_set(acc)

        assert(self.lyap.cmax >= 3.)
        assert(self.lyap.cmax <= 3 + acc)


@pytest.mark.skip("needs fixing for tensorflow.")
class LyapunovContinuousTest(unittest.TestCase):
    """Test Continuous-time Lyapunov functions."""

    def test_init(self):
        """Test the initialization."""
        discretization = np.array([1, 2, 3])
        lyap_fun = DeterministicFunction.from_callable(
            lambda x: np.ones((3, 1)),
            lambda x: np.ones((3, 1)) * 0.5)

        dynamics = mock.create_autospec(DeterministicFunction)
        l = 0.3
        eps = 0.5

        lyap = LyapunovContinuous(discretization, lyap_fun, dynamics, 0.3, 0.5)
        assert_allclose(lyap.threshold, -l * eps)

        dynamics = np.array([[1, 2, 3]]).T
        a1, a2 = lyap.v_decrease_confidence(lyap.discretization, dynamics)
        assert(a2 == 0)
        true_mean = true_error = 0.5 * dynamics.squeeze()
        assert_allclose(a1, true_mean)

        a1, a2 = lyap.v_decrease_confidence(lyap.discretization,
                                            (dynamics, dynamics))
        assert_allclose(a1, true_mean)
        assert_allclose(a2, true_error)

    def test_lipschitz_constant(self):
        """Test the Lipschitz constant that is returned."""
        a = LyapunovContinuous.lipschitz_constant(1, 2, 3, 4)
        assert_allclose(a, 10)


class LyapunovDiscreteTest(unittest.TestCase):
    """Test Continuous-time Lyapunov functions."""

    def test_init(self):
        """Test the initialization."""
        discretization = np.array([1, 2, 3])
        lyap_fun = lambda x: np.arange(3)[:, None]

        dynamics = mock.create_autospec(DeterministicFunction)
        lf = 0.4
        lv = 0.3
        eps = 0.5

        lyap = LyapunovDiscrete(discretization, lyap_fun, dynamics, lf, lv,
                                eps)
        assert_allclose(lyap.threshold, -lv * (1 + lf) * eps)

        dynamics = np.array([[1, 2, 3]]).T
        a1, a2 = lyap.v_decrease_confidence(None, dynamics)
        assert(a2 == 0)
        true_mean = np.zeros(3)
        true_error = lv * (np.arange(3) + 1)
        assert_allclose(a1, true_mean)

        a1, a2 = lyap.v_decrease_confidence(None, (dynamics, dynamics))
        assert_allclose(a1, true_mean)
        assert_allclose(a2, true_error)


def test_smallest_boundary_value():
    """Test the boundary value function."""
    with tf.Session() as sess:
        fun = lambda x: 2 * tf.reduce_sum(tf.abs(x), axis=1)
        discretization = GridWorld([[-1.5, 1], [-1, 1.5]], [3, 3])
        min_value = smallest_boundary_value(fun, discretization)
        assert min_value == 2.5


if __name__ == '__main__':
    unittest.main()
