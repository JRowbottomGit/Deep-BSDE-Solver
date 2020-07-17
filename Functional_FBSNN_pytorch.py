import numpy as np
import matplotlib.pyplot as plt
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim

def weights_init(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)

#need to vmap and jit it
def forward(model, t, X):
    input = torch.cat((t, X), 1)
    u = model(input)  # M x 1
    return u

def grad_forward(model, t, X):
    input = torch.cat((t, X), 1)
    u = model(input)  # M x 1
    Du = torch.autograd.grad(outputs=[u], inputs=[X], grad_outputs=torch.ones_like(u), allow_unused=True,
                             retain_graph=True, create_graph=True)[0]
    return Du

# def Dg_tf(X):  # M x D
#
#     g = g_tf(X)
#     Dg = torch.autograd.grad(outputs=[g], inputs=[X], grad_outputs=torch.ones_like(g), allow_unused=True,
#                              retain_graph=True, create_graph=True)[0]  # M x D
#     return Dg

def XYpaths(model, t, W, Xzero):
    Xzero = torch.from_numpy(Xzero).float().to(device)  # initial point
    Xzero.requires_grad = True
    t = torch.from_numpy(t).float().to(device)  # initial point
    t.requires_grad = True
    W = torch.from_numpy(W).float().to(device)  # initial point
    W.requires_grad = True

    X_list = []
    Y_list = []
    Y_tilde_list = []

    t0 = t[:, 0, :]
    W0 = W[:, 0, :]

    X0 = Xzero.repeat(M, 1).reshape((M, D))  # M x D repeat numpy function
    Y0 = forward(model, t0, X0)
    Z0 = grad_forward(model, t0, X0)  # M x 1, M x D

    X_list.append(X0)
    Y_list.append(Y0)

    for n in range(0, N):
        t1 = t[:, n + 1, :]
        W1 = W[:, n + 1, :]
        X1 = X0 + mu_tf(t0, X0, Y0, Z0) * (t1 - t0) + torch.squeeze(
            torch.matmul(sigma_tf(t0, X0, Y0), (W1 - W0).unsqueeze(-1)), dim=-1)
        Y1_tilde = Y0 + phi_tf(t0, X0, Y0, Z0) * (t1 - t0) + torch.sum(
            Z0 * torch.squeeze(torch.matmul(sigma_tf(t0, X0, Y0), (W1 - W0).unsqueeze(-1))), dim=1,
            keepdim=True)

        Y1 = forward(model, t1, X1)
        Z1 = grad_forward(model, t1, X1)  # M x 1, M x D

        t0 = t1
        W0 = W1
        X0 = X1
        Y0 = Y1
        Z0 = Z1

        X_list.append(X0)
        Y_list.append(Y0)
        Y_tilde_list.append(Y1_tilde)

    X = torch.stack(X_list, dim=1)
    Y = torch.stack(Y_list, dim=1)
    Y_tilde = torch.stack(Y_tilde_list, dim=1)

    return X, Y, Y_tilde

# jit static arg nums (0,1,2,3,)
def loss_function(model, M, D, N, t, W, Xzero):   #idea take M,D out by making X0
    loss = 0
    X,Y,Y_tilde = XYpaths(model, t, W, Xzero)

    squared = torch.pow(Y[:, 1:, :] - Y_tilde, 2)     # Y is 51, Y_tilde is 50

    loss += torch.sum(torch.pow(Y[:, 1:, :] - Y_tilde, 2))

    loss += torch.sum(torch.pow(Y[:,N,:] - g_tf(X[:,N,:]), 2)) #terminal condition

    # loss += torch.sum(torch.pow(Z1 - Dg_tf(X1), 2)) #terminal 1st order condition remove for now
    return loss

def fetch_minibatch(T,M,N,D):  # Generate time + a Brownian motion

    Dt = np.zeros((M, N + 1, 1))  # M x (N+1) x 1
    DW = np.zeros((M, N + 1, D))  # M x (N+1) x D

    dt = T / N

    Dt[:, 1:, :] = dt
    DW[:, 1:, :] = np.sqrt(dt) * np.random.normal(size=(M, N, D))

    t = np.cumsum(Dt, axis=1)  # M x (N+1) x 1
    W = np.cumsum(DW, axis=1)  # M x (N+1) x D
    # t = torch.from_numpy(t).float().to(self.device) <- cancel these out so stays as numpy
    # W = torch.from_numpy(W).float().to(self.device) <- cancel these out so stays as numpy
    return t, W

def train(T,M,N,D, model, N_Iter, learning_rate, Xzero):
    # Record the loss
    training_loss = []
    iteration = []

    loss_temp = np.array([])

    previous_it = 0
    if iteration != []:
        previous_it = iteration[-1]

    # Optimizers
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    start_time = time.time()
    for it in range(previous_it, previous_it + N_Iter):
        optimizer.zero_grad()
        t_batch, W_batch = fetch_minibatch(T,M,N,D)  # M x (N+1) x 1, M x (N+1) x D

        loss = loss_function(model, M, D, N, t_batch, W_batch, Xzero)   #idea take M,D out by making X0

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_temp = np.append(loss_temp, loss.cpu().detach().numpy())

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
            loss_temp = np.array([])

            iteration.append(it)

        graph = np.stack((iteration, training_loss))
    return graph

def phi_tf(t, X, Y, Z):  # M x 1, M x D, M x 1, M x D
        return 0.05 * (Y - torch.sum(X * Z, dim=1, keepdim=True))  # M x 1

def g_tf(X):  # M x D
    return torch.sum(X ** 2, 1, keepdim=True)  # M x 1

def mu_tf(t, X, Y, Z):  # M x 1, M x D, M x 1, M x D
    return torch.zeros([M, D]).to(device)  # M x D

def sigma_tf(t, X, Y):  # M x 1, M x D, M x 1
    return 0.4 * torch.diag_embed(X)  # M x D x D

def u_exact(t, X):  # (N+1) x 1, (N+1) x D
    r = 0.05
    sigma_max = 0.4
    return np.exp((r + sigma_max ** 2) * (T - t)) * np.sum(X ** 2, 1, keepdims=True)  # (N+1) x 1

def run_model(T,M,N,D, model, Xzero, N_Iter, learning_rate):

    tot = time.time()
    samples = 5
    graph = train(T,M,N,D, model, N_Iter, learning_rate, Xzero)
    print("total time:", time.time() - tot, "s")

    np.random.seed(42)
    t_test, W_test = fetch_minibatch(T,M,N,D)

    X_pred, Y_pred, Y_tilde_pred = XYpaths(model, t_test, W_test, Xzero)

    if type(t_test).__module__ != 'numpy':
        t_test = t_test.cpu().numpy()
    if type(X_pred).__module__ != 'numpy':
        X_pred = X_pred.cpu().detach().numpy()
    if type(Y_pred).__module__ != 'numpy':
        Y_pred = Y_pred.cpu().detach().numpy()

    Y_test = np.reshape(u_exact(np.reshape(t_test[0:M, :, :], [-1, 1]), np.reshape(X_pred[0:M, :, :], [-1, D])),
                        [M, -1, 1])

    plt.figure()
    plt.plot(graph[0], graph[1])
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.yscale("log")
    plt.title('Evolution of the training loss')

    plt.figure()
    plt.plot(t_test[0:1, :, 0].T, Y_pred[0:1, :, 0].T, 'b', label='Learned $u(t,X_t)$')
    plt.plot(t_test[0:1, :, 0].T, Y_test[0:1, :, 0].T, 'r--', label='Exact $u(t,X_t)$')
    plt.plot(t_test[0:1, -1, 0], Y_test[0:1, -1, 0], 'ko', label='$Y_T = u(T,X_T)$')

    plt.plot(t_test[1:samples, :, 0].T, Y_pred[1:samples, :, 0].T, 'b')
    plt.plot(t_test[1:samples, :, 0].T, Y_test[1:samples, :, 0].T, 'r--')
    plt.plot(t_test[1:samples, -1, 0], Y_test[1:samples, -1, 0], 'ko')

    plt.plot([0], Y_test[0, 0, 0], 'ks', label='$Y_0 = u(0,X_0)$')

    plt.xlabel('$t$')
    plt.ylabel('$Y_t = u(t,X_t)$')
    plt.title(str(D) + '-dimensional Black-Scholes-Barenblatt, ' + mode + "-" + activation + "_JR")
    plt.legend()

    errors = np.sqrt((Y_test - Y_pred) ** 2 / Y_test ** 2)
    mean_errors = np.mean(errors, 0)
    std_errors = np.std(errors, 0)

    plt.figure()
    plt.plot(t_test[0, :, 0], mean_errors, 'b', label='mean')
    plt.plot(t_test[0, :, 0], mean_errors + 2 * std_errors, 'r--', label='mean + two standard deviations')
    plt.xlabel('$t$')
    plt.ylabel('relative error')
    plt.title(str(D) + '-dimensional-Black-Scholes-Barenblatt-' + mode + "-" + activation + "_JR")
    plt.legend()
    plt.savefig(str(D) + '-dimensional-Black-Scholes-Barenblatt-' + mode + "-" + activation + "_JR")
    cwd = os.getcwd()
    print(cwd)

    text_file = open("JR_Output.txt", "w")
    text_file.write(f"where is this file\nhere: {cwd}")
    text_file.close()

if __name__ == "__main__":
    tot = time.time()
    M = 100  # number of trajectories (batch size)
    N = 50  # number of time snapshots
    D = 100  # number of dimensions

    layers = [D + 1] + 4 * [256] + [1]

    Xzero = np.array([1.0, 0.5] * int(D / 2))[None, :]

    T = 1.0

    "Available architectures"
    mode = "FC"  # FC, Resnet and NAIS-Net are available
    activation = "ReLU"  # Sine and ReLU are available
    if activation == "ReLU":
        activation_function = nn.ReLU()
    device_idx = 0
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")

    model_layers = []
    for i in range(len(layers) - 2):  #-1 ????
        model_layers.append(nn.Linear(in_features=layers[i], out_features=layers[i + 1]))
        model_layers.append(activation_function)
    model_layers.append(nn.Linear(in_features=layers[-2], out_features=layers[-1]))

    print(f"device {device}")
    model = nn.Sequential(*model_layers).to(device)
    model.apply(weights_init)

    run_model(T,M,N,D, model, Xzero, 2 * 10 ** 2, 1e-3)
    # run_model(model, 2*10**4, 1e-3)