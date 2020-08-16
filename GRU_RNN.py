# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

import math
import time
import numpy as onp
import jax.numpy as np
from jax import grad, jit, vmap, value_and_grad
from jax import random
from jax.experimental import stax
from jax.experimental.stax import (BatchNorm, Conv, Dense, Flatten,
                                   Relu, LogSoftmax)
from jax.experimental import optimizers


# Generate key which is used to generate random numbers
key = random.PRNGKey(1)

import numpy as onp
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(context='poster', style='white',
        font='sans-serif', font_scale=1, color_codes=True, rc=None)

def generate_ou_process(batch_size, num_dims, mu, tau, sigma, noise_std, dt = 0.1):
    """ Ornstein-Uhlenbeck process sequences to train on """
    ou_x = onp.zeros((batch_size, num_dims))
    ou_x[:, 0] = onp.random.random(batch_size)
    for t in range(0, num_dims):
        dx = -(ou_x[:, t-1]-mu)/tau * dt + sigma*onp.sqrt(2/tau)*onp.random.normal(0, 1, batch_size)*onp.sqrt(dt)
        ou_x[:, t] =  ou_x[:, t-1] + dx

    ou_x_noise = ou_x + onp.random.multivariate_normal(onp.zeros(num_dims),
                                                        noise_std*onp.eye(num_dims),
                                                        batch_size)

    return ou_x, ou_x_noise

def plot_ou_process(x, x_tilde=None, x_pred=None,
                   title=r"Ornstein-Uhlenbeck Process"):
    """ Visualize an example datapoint (OU process or convolved noise)"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(range(len(x)), x, label="Ground Truth", alpha=0.75)
    if x_tilde is not None:
        ax.plot(range(len(x_tilde)), x_tilde, label="Noisy", alpha=0.75)
    if x_pred is not None:
        ax.plot(range(len(x_pred)), x_pred, label="Prediction")
    ax.set_ylabel(r"OU Process")
    ax.set_xlabel(r"Time $t$")
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=12)
    return

def plot_ou_loss(train_loss, title="Train Loss - OU GRU-RNN"):
    """ Visualize the learning performance of the OU process RNN """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(train_loss)
    ax.set_xlabel("# Batch Updates")
    ax.set_ylabel("Batch Loss")
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Generate & plot a time series generated by the OU process
    x_0, mu, tau, sigma, dt = 0, 1, 2, 0.5, 0.1
    noise_std = 0.01
    num_dims, batch_size = 100, 64  # Number of timesteps in process

    x, x_tilde = generate_ou_process(batch_size, num_dims, mu, tau,
                                     sigma, noise_std, dt)
    plot_ou_process(x[0, :], x_tilde[0, :])

# Generate & plot a time series generated by the OU process
x_0, mu, tau, sigma, dt = 0, 1, 2, 0.5, 0.1
noise_std = 0.01
num_dims, batch_size = 100, 64  # Number of timesteps in process

x, x_tilde = generate_ou_process(batch_size, num_dims, mu, tau,
                                 sigma, noise_std, dt)
plot_ou_process(x[0, :], x_tilde[0, :])

from jax.nn import sigmoid
from jax.nn.initializers import glorot_normal, normal

from functools import partial
from jax import lax

def GRU(out_dim, W_init=glorot_normal(), b_init=normal()): ### out_dim = num_hidden_units = 12
    def init_fun(rng, input_shape):                     ### (batch_size=64, num_dims=10#0, 1)
        """ Initialize the GRU layer for stax """
        hidden = b_init(rng, (input_shape[0], out_dim))  ### this is the H0 initial guess, that's why is dependent on batch size

        k1, k2, k3 = random.split(rng, num=3)
        reset_W, reset_U, reset_b = (
            W_init(k1, (input_shape[2], out_dim)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),)

        k1, k2, k3 = random.split(rng, num=3)
        update_W, update_U, update_b = (
            W_init(k1, (input_shape[2], out_dim)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),)

        k1, k2, k3 = random.split(rng, num=3)
        out_W, out_U, out_b = (
            W_init(k1, (input_shape[2], out_dim)),
            W_init(k2, (out_dim, out_dim)),
            b_init(k3, (out_dim,)),)
        # Input dim 0 represents the batch dimension
        # Input dim 1 represents the time dimension (before scan moveaxis)
        output_shape = (input_shape[0], input_shape[1], out_dim)
        return (output_shape,
            (hidden,
             (update_W, update_U, update_b),
             (reset_W, reset_U, reset_b),
             (out_W, out_U, out_b),),)

    def apply_fun(params, inputs, **kwargs):
        """ Loop over the time steps of the input sequence """
        h = params[0]

        def apply_fun_scan(params, hidden, inp):
            """ Perform single step update of the network """
            _, (update_W, update_U, update_b), (reset_W, reset_U, reset_b), (
                out_W, out_U, out_b) = params

            reset_gate = sigmoid(np.dot(inp, reset_W) +
                                 np.dot(hidden, reset_U) + reset_b)
            update_gate = sigmoid(np.dot(inp, update_W) +
                                  np.dot(hidden, update_U) + update_b)
            output_gate = np.tanh(np.dot(inp, out_W)
                                  + np.dot(np.multiply(reset_gate, hidden), out_U)
                                  + out_b)
            output = np.multiply(update_gate, hidden) + np.multiply(1-update_gate, output_gate)
            hidden = output
            return hidden, hidden

        # Move the time dimension to position 0 so lax.scan can loop over time
        inputs = np.moveaxis(inputs, 1, 0)
        f = partial(apply_fun_scan, params) # We use partial to “clone” all the params to use at all timesteps.
        _, out = lax.scan(f, h, inputs)
        return out

    return init_fun, apply_fun

num_dims = 10#0              # Number of OU timesteps
batch_size = 64            # Batchsize
num_hidden_units = 12      # GRU cells in the RNN layer

# Initialize the network and perform a forward pass
init_fun, gru_rnn = stax.serial(Dense(num_hidden_units), Relu,
                                GRU(num_hidden_units), Dense(1))      #<-this Dense(1) is applied to every lax.scan output from the GRU loop??? hence shape of pred
_, params = init_fun(key, (batch_size, num_dims, 1))

def mse_loss(params, inputs, targets):
    """ Calculate the Mean Squared Error Prediction Loss. """
    preds = gru_rnn(params, inputs)
    return np.mean((preds - targets)**2)

@jit
def update(params, x, y, opt_state):
    """ Perform a forward pass, calculate the MSE & perform a SGD step. """
    loss, grads = value_and_grad(mse_loss)(params, x, y)
    opt_state = opt_update(0, grads, opt_state)
    return get_params(opt_state), opt_state, loss

step_size = 1e-4
opt_init, opt_update, get_params = optimizers.adam(step_size)
opt_state = opt_init(params)

num_batches = 1000

train_loss_log = []
start_time = time.time()
for batch_idx in range(num_batches):
    x, x_tilde = generate_ou_process(batch_size, num_dims, mu, tau, sigma, noise_std)
    x_in = np.expand_dims(x_tilde[:, :(num_dims-1)], 2)
    y = np.array(x[:, 1:])
    params, opt_state, loss = update(params, x_in, y, opt_state)
    batch_time = time.time() - start_time
    train_loss_log.append(loss)

    if batch_idx % 100 == 0:
        start_time = time.time()
        print("Batch {} | T: {:0.2f} | MSE: {:0.2f} |".format(batch_idx, batch_time, loss))

plot_ou_loss(train_loss_log)

# Plot a prediction and ground truth OU process
x_true, x_tilde = generate_ou_process(batch_size, num_dims, mu, tau, sigma, noise_std)
x = np.expand_dims(x_tilde[:, :(num_dims-1)], 2)
y = np.array(x[:, 1:])
preds = gru_rnn(params, x)

x = onp.array(x)
y = onp.array(y)
x_pred = onp.array(preds)
plot_ou_process(x_true[:, 0], x_tilde=x_tilde[:, 0], x_pred=x_pred[0, :],
               title=r"Ornstein-Uhlenbeck Prediction")
