import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import time
import os
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.ops import index, index_add, index_update

def init_random_params(scale, layer_sizes, rng=npr.RandomState(0)):
  return [(scale * rng.randn(m, n), scale * rng.randn(n))
          for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])]
def relu(x):
    return jnp.maximum(0, x)

#need to vmap and jit it
def forward(params, t, X):
    # print(t)
    # print(X)
    # print(f"t.shape {t.shape}")
    # print(f"X.shape) {X.shape}")
    input = jnp.concatenate((t, X), 0)  # M x D+1
    # print("here?")
    activations = input
    for w, b in params[:-1]:
        outputs = jnp.dot(activations, w) + b
        activations = relu(outputs)

    final_w, final_b = params[-1]
    u = jnp.dot(activations, final_w) + final_b
    return jnp.reshape(u,())  #need scalar for grad

def forward2(params, t, X):
    # print(t)
    # print(X)
    # print(f"t.shape {t.shape}")
    # print(f"X.shape) {X.shape}")
    input = jnp.concatenate((t, X), 0)  # M x D+1
    # print("here?")
    activations = input
    counter = 0
    for w, b in params[:-1]:
        outputs = jnp.dot(activations, w) + b
        activations = relu(outputs)
        counter += 1
    final_w, final_b = params[-1]
    u = jnp.dot(activations, final_w) + final_b
    return jnp.reshape(u,())  #need scalar for grad

vforward = vmap(forward, in_axes=(None, 0, 0))

def grad_forward(params, t, X):
    gradu = grad(forward,argnums=(2)) #<wrt X only not params or t
    # partial(grad(loss), params)
    Du = gradu(params, t, X)
    return Du

vgrad_forward = vmap(grad_forward, in_axes=(None, 0, 0))

# def Dg_tf(X):  # M x D
#
#     g = g_tf(X)
#     Dg = torch.autograd.grad(outputs=[g], inputs=[X], grad_outputs=torch.ones_like(g), allow_unused=True,
#                              retain_graph=True, create_graph=True)[0]  # M x D
#     return Dg

def fetch_minibatch(T,M,N,D):  # Generate time + a Brownian motion

    Dt = jnp.zeros((M, N + 1, 1))  # M x (N+1) x 1
    DW = jnp.zeros((M, N + 1, D))  # M x (N+1) x D

    dt = T / N
    new_Dt = index_update(Dt, index[:, 1:, :], dt)
    new_dw = jnp.sqrt(dt) * np.random.normal(size=(M, N, D))   ###fix this to jnp random
    new_DW = index_update(DW, index[:, 1:, :], new_dw)

    t = jnp.cumsum(new_Dt, axis=1)  # M x (N+1) x 1
    W = jnp.cumsum(new_DW, axis=1)  # M x (N+1) x D
    return t, W

def XYpaths(params, t, W, X0):

    X_list = []
    Y_list = []
    Y_tilde_list = []

    t0 = t[0, :]
    W0 = W[0, :]

    # X0 = Xzero.repeat(M, 1).reshape((M, D))  # M x D repeat numpy function
    # print(t.shape)
    # print(W.shape)
    # print(W0.shape)
    # print(X0.shape)

    # Y0 = forward(params, t0, X0)
    Y0 = jnp.asarray([forward(params, t0, X0)]) #need to array it up as make scalar for grad
    # print("Y0")
    # print(Y0.shape)
    Z0 = grad_forward(params, t0, X0)

    X_list.append(X0)
    Y_list.append(Y0)

    for n in range(0, N):
        t1 = t[n + 1, :]
        W1 = W[n + 1, :]

        X1 = X0 + mu_tf(t0, X0, Y0, Z0) * (t1 - t0) + jnp.dot(sigma_tf(t0, X0, Y0), (W1 - W0))
        # print("here3")
        # print(X0.shape)
        # print(Y0.shape)
        # print(Z0.shape)
        # print(X1.shape)
        # print(f"mu_tf(t0, X0, Y0, Z0).shape {mu_tf(t0, X0, Y0, Z0).shape}")
        # print(f"sigma_tf(t0, X0, Y0).shape {sigma_tf(t0, X0, Y0).shape}")

        Y1_tilde = Y0 + phi_tf(t0, X0, Y0, Z0) * (t1 - t0) + jnp.dot(jnp.dot(Z0.T, sigma_tf(t0, X0, Y0)),
                                                                          (W1 - W0))
        # print("here4")
        # print(t1.shape)
        # print(X1.shape)
        # Y1 = forward(params, t1, X1)
        Y1 = jnp.asarray([forward(params, t1, X1)])
        # print("here5")
        Z1 = grad_forward(params, t1, X1)
        # print("here6")

        t0 = t1
        W0 = W1
        X0 = X1
        Y0 = Y1
        Z0 = Z1

        X_list.append(X0)
        Y_list.append(Y0)
        Y_tilde_list.append(Y1_tilde)

    X = jnp.stack(X_list, axis=1)
    Y = jnp.stack(Y_list, axis=1)
    Y_tilde = jnp.stack(Y_tilde_list, axis=1)

    return X, Y, Y_tilde

vXYpaths = vmap(XYpaths, in_axes=(None, 0,0,None))

def XYpaths2(params, t, W, X0):

    X_list = []
    Y_list = []
    Y_tilde_list = []

    t0 = t[0, :]
    W0 = W[0, :]

    # X0 = Xzero.repeat(M, 1).reshape((M, D))  # M x D repeat numpy function
    # print(t.shape)
    # print(W.shape)
    # print(W0.shape)
    # print(X0.shape)

    # Y0 = forward(params, t0, X0)
    Y0 = jnp.asarray([forward2(params, t0, X0)]) #need to array it up as make scalar for grad
    # print("Y0")
    # print(Y0.shape)
    Z0 = grad_forward(params, t0, X0)

    X_list.append(X0)
    Y_list.append(Y0)

    for n in range(0, N):
        t1 = t[n + 1, :]
        W1 = W[n + 1, :]

        X1 = X0 + mu_tf(t0, X0, Y0, Z0) * (t1 - t0) + jnp.dot(sigma_tf(t0, X0, Y0), (W1 - W0))
        # print("here3")
        # print(X0.shape)
        # print(Y0.shape)
        # print(Z0.shape)
        # print(X1.shape)
        # print(f"mu_tf(t0, X0, Y0, Z0).shape {mu_tf(t0, X0, Y0, Z0).shape}")
        # print(f"sigma_tf(t0, X0, Y0).shape {sigma_tf(t0, X0, Y0).shape}")

        Y1_tilde = Y0 + phi_tf(t0, X0, Y0, Z0) * (t1 - t0) + jnp.dot(jnp.dot(Z0.T, sigma_tf(t0, X0, Y0)),
                                                                          (W1 - W0))
        # print("here4")
        # print(t1.shape)
        # print(X1.shape)
        # Y1 = forward(params, t1, X1)
        Y1 = jnp.asarray([forward(params, t1, X1)])
        # print("here5")
        Z1 = grad_forward(params, t1, X1)
        # print("here6")

        t0 = t1
        W0 = W1
        X0 = X1
        Y0 = Y1
        Z0 = Z1

        X_list.append(X0)
        Y_list.append(Y0)
        Y_tilde_list.append(Y1_tilde)

    X = jnp.stack(X_list, axis=1)
    Y = jnp.stack(Y_list, axis=1)
    Y_tilde = jnp.stack(Y_tilde_list, axis=1)

    return X, Y, Y_tilde

vXYpaths2 = vmap(XYpaths2, in_axes=(None, 0,0,None))

# jit static arg nums (0,1,2,3,)
def loss_function(params, t, W, Xzero):   #idea take M,D out by making X0
    loss = 0
    X,Y,Y_tilde = vXYpaths(params, t, W, Xzero)

    loss += jnp.sum(jnp.power(Y[:, :, 1:] - Y_tilde, 2)) # Y is 51, Y_tilde is 50

    loss += jnp.sum(jnp.power(Y[:,:,N] - g_tf(X[:,:,N]), 2)) #terminal condition

    # loss += torch.sum(torch.pow(Z1 - Dg_tf(X1), 2)) #terminal 1st order condition remove for now
    return loss

# @jit
def update(params, t, W, X0):
    # print("here7")
    grads = grad(loss_function,argnums=0)(params, t, W, X0)
    # print("here8")
    return_list = [(w - step_size * dw, b - step_size * db)
          for (w, b), (dw, db) in zip(params, grads)]
    # print("here9")
    return return_list

def train(T,M,N,D, params, N_Iter, learning_rate, Xzero):
    # Record the loss
    training_loss = []
    iteration = []

    loss_temp = jnp.array([])

    previous_it = 0
    if iteration != []:
        previous_it = iteration[-1]
    # Optimizers
    # optimizer = optim.Adam(params, lr=learning_rate)

    start_time = time.time()
    for it in range(previous_it, previous_it + N_Iter):
        print(f"Iteration {it}")
        t_batch, W_batch = fetch_minibatch(T,M,N,D)  # M x (N+1) x 1, M x (N+1) x D

        #idea take M,D out by making X0 - probs can just remove from arguments
        loss = loss_function(params, t_batch, W_batch, Xzero)  #annoying as it gets calc'd twice
        print("loss")
        print(loss)
        params = update(params, t_batch, W_batch, Xzero)
        # params = params
        print("update")
        loss_temp = jnp.append(loss_temp, loss)

        # Print
        if it % 100 == 0:
            elapsed = time.time() - start_time
            # print('It: %d, Loss: %.3e, Y0: %.3f, Time: %.2f, Learning Rate: %.3e' %
                  # (it, loss, Y0_pred, elapsed, learning_rate))
            print('It: %d, Loss: %.3e, Time: %.2f, Learning Rate: %.3e' %
                  (it, loss, elapsed, learning_rate))
            start_time = time.time()

        # Loss
        if it % 100 == 0:
            training_loss.append(loss_temp.mean())
            loss_temp = jnp.array([])

            iteration.append(it)

        graph = np.stack((iteration, training_loss))

    loss2 = loss_function(params, t_batch, W_batch, Xzero)  # annoying as it gets calc'd twice
    print("loss2")
    print(loss2)

    return params, graph

def phi_tf(t, X, Y, Z):  # M x 1, M x D, M x 1, M x D
    # print(X.shape)
    # print(Y.shape)
    # print(Z.shape)
    # print("here2")
    # return 0.05 * (Y - jnp.dotsum(X * Z, axis=1, keepdims=True))  # M x 1
    return 0.05 * (Y - jnp.dot(X, Z))  # M x 1

def g_tf(X):  # M x D
    return jnp.sum(X ** 2, 1, keepdims=True)  # M x 1

def mu_tf(t, X, Y, Z):  # M x 1, M x D, M x 1, M x D
    return jnp.zeros(D)  # M x D

def sigma_tf(t, X, Y):  # M x 1, M x D, M x 1
    # print(f"sigX {X.shape}")
    # print(f"jnp.diag(X) {jnp.diag(X).shape}")
    # return 0.4 * torch.diag_embed(X)  # M x D x D
    return 0.4 * jnp.diag(X)  # M x D x D
vsigma_tf = vmap(sigma_tf, in_axes=0)

def u_exact(t, X):  # (N+1) x 1, (N+1) x D
    r = 0.05
    sigma_max = 0.4
    return jnp.exp((r + sigma_max ** 2) * (T - t)) * jnp.sum(X ** 2, 1, keepdims=True)  # (N+1) x 1

def run_model(T,M,N,D, init_params, Xzero, N_Iter, learning_rate):

    tot = time.time()
    samples = 5
    params, graph = train(T,M,N,D, init_params, N_Iter, learning_rate, Xzero)
    print("total time:", time.time() - tot, "s")

    np.random.seed(42)
    t_test, W_test = fetch_minibatch(T,M,N,D)

    loss3 = loss_function(params, t_test, W_test, Xzero)  # annoying as it gets calc'd twice
    print("loss3")
    print(loss3)

    X_pred, Y_pred, Y_tilde_pred = vXYpaths2(params, t_test, W_test, Xzero)

    # Y_test = jnp.reshape(u_exact(np.reshape(t_test[0:M, :, :], [-1, 1]), jnp.reshape(X_pred[0:M, :, :], [-1, D])),
    #                     [M, -1, 1])   #fix all these uneccessary reshapes at some point
    Y_test = jnp.reshape(u_exact(np.reshape(t_test[0:M, :, :], [-1, 1]), jnp.reshape(X_pred[0:M, :, :], [-1, D])),
                        [M, 1, -1])   #fix all these uneccessary reshapes at some point

    plt.figure()
    plt.plot(graph[0], graph[1])
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.yscale("log")
    plt.title('Evolution of the training loss')

    plt.figure()
    # plt.plot(t_test[0:1, :, 0].T, Y_pred[0:1, :, 0].T, 'b', label='Learned $u(t,X_t)$')  #<-fucked the dimensions of just Y_pred somewhere....
    # plt.plot(t_test[0:1, :, 0].T, Y_test[0:1, :, 0].T, 'r--', label='Exact $u(t,X_t)$')
    # plt.plot(t_test[0:1, -1, 0], Y_test[0:1, -1, 0], 'ko', label='$Y_T = u(T,X_T)$')

    plt.plot(t_test[0:1, :, 0].T, Y_pred[0:1, 0, :].T, 'b', label='Learned $u(t,X_t)$')
    plt.plot(t_test[0:1, :, 0].T, Y_test[0:1, 0, :].T, 'r--', label='Exact $u(t,X_t)$')
    plt.plot(t_test[0:1, -1, 0], Y_test[0:1, 0, -1], 'ko', label='$Y_T = u(T,X_T)$')

    # plt.plot(t_test[1:samples, :, 0].T, Y_pred[1:samples, :, 0].T, 'b')
    plt.plot(t_test[1:samples, :, 0].T, Y_pred[1:samples, 0, :].T, 'b')

    # plt.plot(t_test[1:samples, :, 0].T, Y_test[1:samples, :, 0].T, 'r--')
    # plt.plot(t_test[1:samples, -1, 0], Y_test[1:samples, -1, 0], 'ko')
    plt.plot(t_test[1:samples, :, 0].T, Y_test[1:samples, 0, :].T, 'r--')
    plt.plot(t_test[1:samples, -1, 0], Y_test[1:samples, 0, -1], 'ko')

    plt.plot([0], Y_test[0, 0, 0], 'ks', label='$Y_0 = u(0,X_0)$')

    plt.xlabel('$t$')
    plt.ylabel('$Y_t = u(t,X_t)$')
    plt.title(str(D) + '-dimensional Black-Scholes-Barenblatt, ' + "FC" + "-" + "ReLu" + "_JRJaxvec")
    plt.legend()

    errors = jnp.sqrt((Y_test - Y_pred) ** 2 / Y_test ** 2)
    # mean_errors = jnp.mean(errors, 0)
    # std_errors = jnp.std(errors, 0)
    mean_errors = jnp.mean(errors, 0)[0,:]
    std_errors = jnp.std(errors, 0)[0,:]

    plt.figure()
    # plt.plot(t_test[0, :, 0], mean_errors, 'b', label='mean')
    # plt.plot(t_test[0, :, 0], mean_errors + 2 * std_errors, 'r--', label='mean + two standard deviations')
    plt.plot(t_test[0, :, 0], mean_errors, 'b', label='mean')
    plt.plot(t_test[0, :, 0], mean_errors + 2 * std_errors, 'r--', label='mean + two standard deviations')

    plt.xlabel('$t$')
    plt.ylabel('relative error')
    plt.title(str(D) + '-dimensional-Black-Scholes-Barenblatt-' + "FC" + "-" + "ReLu" + "_JRJaxVec")
    plt.legend()
    plt.savefig(str(D) + '-dimensional-Black-Scholes-Barenblatt-' + "FC" + "-" + "ReLu" + "_JRJaxVec")
    cwd = os.getcwd()
    print(cwd)

    text_file = open("JRJaxVec_Output.txt", "w")
    text_file.write(f"where is this file\nhere: {cwd}")
    text_file.close()

if __name__ == "__main__":
    tot = time.time()
    M = 98  # number of trajectories (batch size)
    N = 50  # number of time snapshots
    D = 100  # number of dimensions
    T = 1.0
    # step_size = 0.001
    step_size = 0.000001

    layers = [D + 1] + 4 * [256] + [1]  #[101, 256, 256, 256, 256, 1]

    param_scale = 0.1
    params = init_random_params(param_scale, layers)

    Xzero = jnp.array([1.0, 0.5] * int(D / 2))#[None, :]   ###WHY????? 1 0.5 1 0.5 1....

    # run_model(T,M,N,D, params, Xzero, 2 * 10 **2, 1e-3)
    # run_model(T,M,N,D, params, Xzero, 2 * 10 **1, 1e-3)
    run_model(T,M,N,D, params, Xzero, 200, step_size)

    # run_model(model, 2*10**4, 1e-3)