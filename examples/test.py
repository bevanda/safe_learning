from __future__ import division, print_function

from functools import partial

import gpflow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, linalg

# Nice progress bars
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

import safe_learning
from safe_learning import InvertedPendulum

# command for Jupyter Notebooks %matplotlib inline

# Open a new session (close old one if exists)
try:
    session.close()
except NameError:
    pass

session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())

n = 2
m = 1

# 'Wrong' model parameters
mass = 0.1
friction = 0.
length = 0.5
gravity = 9.81
inertia = mass * length ** 2

# True model parameters
true_mass = 0.15
true_friction = 0.1
true_length = length
true_inertia = true_mass * true_length ** 2

# Input saturation
x_max = np.deg2rad(30)
u_max = gravity * true_mass * true_length * np.sin(x_max)

# Normalization
norm_state = np.array([x_max, np.sqrt(gravity / length)])
norm_action = np.array([u_max])

# Corresponding dynamic systems
true_dynamics = InvertedPendulum(mass=true_mass, length=true_length, friction=true_friction,
                                 normalization=(norm_state, norm_action))

wrong_pendulum = InvertedPendulum(mass=mass, length=length, friction=friction,
                                  normalization=(norm_state, norm_action))

# LQR cost matrices
q = 1 * np.diag([1., 2.])
r = 1.2 * np.array([[1]], dtype=safe_learning.config.np_dtype)

# Quadratic (LQR) reward function
gamma = 0.98
reward_function = safe_learning.QuadraticFunction(linalg.block_diag(-q, -r))


# x_min, x_max, discretization\
state_limits = np.array([[-2., 2.], [-1.5, 1.5]])
action_limits = np.array([[-1, 1]])
num_states = [2001, 1501]

safety_disc = safe_learning.GridWorld(state_limits, num_states)
policy_disc = safe_learning.GridWorld(state_limits, [55, 55])

# Discretization constant
tau = np.min(safety_disc.unit_maxes)

print('Grid size: {0}'.format(safety_disc.nindex))

A, B = wrong_pendulum.linearize()
lipschitz_dynamics = 1

noise_var = 0.001 ** 2

m_true = np.hstack((true_dynamics.linearize()))
m = np.hstack((A, B))

variances = (m_true - m) ** 2

# Make sure things remain
np.clip(variances, 1e-5, None, out=variances)

# Kernels
kernel1 = (gpflow.kernels.Linear(3, variance=variances[0, :], ARD=True)
           + gpflow.kernels.Matern32(1, lengthscales=1, active_dims=[0])
           * gpflow.kernels.Linear(1, variance=variances[0, 1]))

kernel2 = (gpflow.kernels.Linear(3, variance=variances[1, :], ARD=True)
           + gpflow.kernels.Matern32(1, lengthscales=1, active_dims=[0])
           * gpflow.kernels.Linear(1, variance=variances[1, 1]))

# Mean dynamics

mean_dynamics = safe_learning.LinearSystem((A, B), name='mean_dynamics')
mean_function1 = safe_learning.LinearSystem((A[[0], :], B[[0], :]), name='mean_dynamics_1')
mean_function2 = safe_learning.LinearSystem((A[[1], :], B[[1], :]), name='mean_dynamics_2')

# Define a GP model over the dynamics
gp1 = gpflow.gpr.GPR(np.empty((0, 3), dtype=safe_learning.config.np_dtype),
                    np.empty((0, 1), dtype=safe_learning.config.np_dtype),
                    kernel1,
                    mean_function=mean_function1)
gp1.likelihood.variance = noise_var

gp2 = gpflow.gpr.GPR(np.empty((0, 3), dtype=safe_learning.config.np_dtype),
                    np.empty((0, 1), dtype=safe_learning.config.np_dtype),
                    kernel2,
                    mean_function=mean_function2)
gp2.likelihood.variance = noise_var

gp1_fun = safe_learning.GaussianProcess(gp1)
gp2_fun = safe_learning.GaussianProcess(gp2)

dynamics = safe_learning.FunctionStack((gp1_fun, gp2_fun))

# Compute the optimal policy for the linear (and wrong) mean dynamics
k, s = safe_learning.utilities.dlqr(A, B, q, r)
init_policy = safe_learning.LinearSystem((-k), name='initial_policy')
init_policy = safe_learning.Saturation(init_policy, -1, 1)

# Define the Lyapunov function corresponding to the initial policy
init_lyapunov = safe_learning.QuadraticFunction(s)

# Define a neural network policy
relu = tf.nn.relu
policy = safe_learning.NeuralNetwork(layers=[32, 32, 1],
                                     nonlinearities=[relu, relu, tf.nn.tanh],
                                     output_scale=action_limits[0, 1])

# Define value function approximation
value_function = safe_learning.Triangulation(policy_disc,
                                             -init_lyapunov(policy_disc.all_points).eval(),
                                             project=True)

# Define policy optimization problem
rl = safe_learning.PolicyIteration(
    policy,
    dynamics,
    reward_function,
    value_function,
    gamma=gamma)

with tf.name_scope('rl_mean_optimization'):
    rl_opt_value_function = rl.optimize_value_function()

    # Placeholder for states
    tf_states_mean = tf.placeholder(safe_learning.config.dtype, [None, 2])

    # Optimize for expected gain
    values = rl.future_values(tf_states_mean)
    policy_loss = -tf.reduce_mean(values)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    adapt_policy_mean = optimizer.minimize(policy_loss, var_list=rl.policy.parameters)

# Define a neural network policy
relu = tf.nn.relu
policy = safe_learning.NeuralNetwork(layers=[32, 32, 1],
                                     nonlinearities=[relu, relu, tf.nn.tanh],
                                     output_scale=action_limits[0, 1])

# Define value function approximation
value_function = safe_learning.Triangulation(policy_disc,
                                             -init_lyapunov(policy_disc.all_points).eval(),
                                             project=True)

# Define policy optimization problem
rl = safe_learning.PolicyIteration(
    policy,
    dynamics,
    reward_function,
    value_function,
    gamma=gamma)

with tf.name_scope('rl_mean_optimization'):
    rl_opt_value_function = rl.optimize_value_function()

    # Placeholder for states
    tf_states_mean = tf.placeholder(safe_learning.config.dtype, [None, 2])

    # Optimize for expected gain
    values = rl.future_values(tf_states_mean)
    policy_loss = -tf.reduce_mean(values)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    adapt_policy_mean = optimizer.minimize(policy_loss, var_list=rl.policy.parameters)


# Start the session
session.run(tf.global_variables_initializer())

for i in tqdm(range(3000)):
    # select random training batches
    rl.feed_dict[tf_states_mean] = policy_disc.sample_continuous(1000)

    session.run(adapt_policy_mean, feed_dict=rl.feed_dict)

    lyapunov_function = -rl.value_function
    lipschitz_lyapunov = lambda x: tf.reduce_max(tf.abs(rl.value_function.gradient(x)),
                                                 axis=1, keepdims=True)

    lipschitz_policy = lambda x: policy.lipschitz()

    a_true, b_true = true_dynamics.linearize()
    lipschitz_dynamics = lambda x: np.max(np.abs(a_true)) + np.max(np.abs(b_true)) * lipschitz_policy(x)

    # Lyapunov function definitial
    lyapunov = safe_learning.Lyapunov(safety_disc,
                                      lyapunov_function,
                                      dynamics,
                                      lipschitz_dynamics,
                                      lipschitz_lyapunov,
                                      tau,
                                      policy=rl.policy,
                                      initial_set=None)

    # Set initial safe set (level set) based on initial Lyapunov candidate
    values = init_lyapunov(safety_disc.all_points).eval()
    cutoff = np.max(values) * 0.005

    lyapunov.initial_safe_set = np.squeeze(values, axis=1) <= cutoff


def plot_safe_set(lyapunov, show=True):
    """Plot the safe set for a given Lyapunov function."""
    plt.imshow(lyapunov.safe_set.reshape(num_states).T,
               origin='lower',
               extent=lyapunov.discretization.limits.ravel(),
               vmin=0,
               vmax=1)

    if isinstance(lyapunov.dynamics, safe_learning.UncertainFunction):
        X = lyapunov.dynamics.functions[0].X
        plt.plot(X[:, 0], X[:, 1], 'rx')

    plt.title('safe set')
    plt.colorbar()
    if show:
        plt.show()


lyapunov.update_safe_set()
plot_safe_set(lyapunov)

with tf.name_scope('policy_optimization'):
    # Placeholder for states
    tf_states = tf.placeholder(safe_learning.config.dtype, [None, 2])

    # Add Lyapunov uncertainty (but only if safety-relevant)
    values = rl.future_values(tf_states, lyapunov=lyapunov)

    policy_loss = -tf.reduce_mean(values)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    adapt_policy = optimizer.minimize(policy_loss, var_list=rl.policy.parameters)


def rl_optimize_policy(num_iter):
    # Optimize value function
    session.run(rl_opt_value_function, feed_dict=rl.feed_dict)

    # select random training batches
    for i in tqdm(range(num_iter)):
        rl.feed_dict[tf_states] = lyapunov.discretization.sample_continuous(1000)
        session.run(adapt_policy, feed_dict=rl.feed_dict)


action_variation = np.array([[-0.02], [0.], [0.02]], dtype=safe_learning.config.np_dtype)

with tf.name_scope('add_new_measurement'):
    action_dim = lyapunov.policy.output_dim
    tf_max_state_action = tf.placeholder(safe_learning.config.dtype,
                                         shape=[1, safety_disc.ndim + action_dim])
    tf_measurement = true_dynamics(tf_max_state_action)


def update_gp():
    """Update the GP model based on an actively selected data point."""
    # Get a new sample location
    max_state_action, _ = safe_learning.get_safe_sample(lyapunov,
                                                        action_variation,
                                                        action_limits,
                                                        num_samples=1000)

    # Obtain a measurement of the true dynamics
    lyapunov.feed_dict[tf_max_state_action] = max_state_action
    measurement = tf_measurement.eval(feed_dict=lyapunov.feed_dict)

    # Add the measurement to our GP dynamics
    lyapunov.dynamics.add_data_point(max_state_action, measurement)

# lyapunov.update_safe_set()
rl_optimize_policy(num_iter=200)
rl_optimize_policy(num_iter=200)

lyapunov.update_safe_set()
plot_safe_set(lyapunov)

for i in range(5):
    print('iteration {} with c_max: {}'.format(i, lyapunov.feed_dict[lyapunov.c_max]))
    for i in tqdm(range(10)):
        update_gp()

    rl_optimize_policy(num_iter=200)
    lyapunov.update_values()

    # Update safe set and plot
    lyapunov.update_safe_set()
    plot_safe_set(lyapunov)


x0 = np.array([[1., -.5]])

states_new, actions_new = safe_learning.utilities.compute_trajectory(true_dynamics, rl.policy, x0, 100)
states_old, actions_old = safe_learning.utilities.compute_trajectory(true_dynamics, init_policy, x0, 100)

t = np.arange(len(states_new)) * true_dynamics.dt

plt.plot(t, states_new[:, 0], label='new')
plt.plot(t, states_old[:, 0], label='old')
plt.xlabel('time [s]')
plt.ylabel('angle [rad]')
plt.legend()
plt.show()

plt.plot(t, states_new[:, 1], label='new')
plt.plot(t, states_old[:, 1], label='old')
plt.xlabel('time [s]')
plt.ylabel('angular velocity [rad/s]')
plt.legend()
plt.show()

plt.plot(t[:-1], actions_new, label='new')
plt.plot(t[:-1], actions_old, label='old')
plt.xlabel('time [s]')
plt.ylabel('actions')
plt.legend()


print('reward old:', tf.reduce_sum(rl.reward_function(states_old[:-1], actions_old)).eval(feed_dict=rl.feed_dict))
print('reward new:', tf.reduce_sum(rl.reward_function(states_new[:-1], actions_new)).eval(feed_dict=rl.feed_dict))