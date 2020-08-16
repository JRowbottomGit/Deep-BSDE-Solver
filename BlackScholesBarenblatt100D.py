import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import os

from FBSNNs import FBSNN


class BlackScholesBarenblatt(FBSNN):
    def __init__(self, Xi, T, M, N, D, layers, mode, activation):
        super().__init__(Xi, T, M, N, D, layers, mode, activation)

    def phi_tf(self, t, X, Y, Z):  # M x 1, M x D, M x 1, M x D
        return 0.05 * (Y - torch.sum(X * Z, dim=1, keepdim=True))  # M x 1

    def g_tf(self, X):  # M x D
        return torch.sum(X ** 2, 1, keepdim=True)  # M x 1

    def mu_tf(self, t, X, Y, Z):  # M x 1, M x D, M x 1, M x D
        return super().mu_tf(t, X, Y, Z)  # M x D

    def sigma_tf(self, t, X, Y):  # M x 1, M x D, M x 1
        return 0.4 * torch.diag_embed(X)  # M x D x D

    ###########################################################################

def u_exact(t, X):  # (N+1) x 1, (N+1) x D
    r = 0.05
    sigma_max = 0.4
    return np.exp((r + sigma_max ** 2) * (T - t)) * np.sum(X ** 2, 1, keepdims=True)  # (N+1) x 1

def run_model(model, N_Iter, learning_rate):
    tot = time.time()

    print(model.device)
    graph = model.train(N_Iter, learning_rate)
    print("total time:", time.time() - tot, "s")

    torch.save(model, "BSB"+ model.mode + "-" + model.activation +"20k")

    np.random.seed(42)
    t_test, W_test = model.fetch_minibatch()
    X_pred, Y_pred = model.predict(Xi, t_test, W_test)

    if type(t_test).__module__ != 'numpy':
        t_test = t_test.cpu().numpy()
    if type(X_pred).__module__ != 'numpy':
        X_pred = X_pred.cpu().detach().numpy()
    if type(Y_pred).__module__ != 'numpy':
        Y_pred = Y_pred.cpu().detach().numpy()

    Y_test = np.reshape(u_exact(np.reshape(t_test[0:M, :, :], [-1, 1]), np.reshape(X_pred[0:M, :, :], [-1, D])),
                        [M, -1, 1])

    # t_test
    # W_test
    # X_pred
    # Y_pred
    # Y_test
    # graph
    if type(W_test).__module__ != 'numpy':
        W_test = W_test.cpu().detach().numpy()
    if type(Y_test).__module__ != 'numpy':
        Y_test = Y_test.cpu().detach().numpy()
    if type(graph).__module__ != 'numpy':
        graph = graph.cpu().detach().numpy()

# #saving things
#     if torch.is_tensor(t_test):
#         t_test = t_test.data.cpu().numpy()
#     else:
#         t_test = np.array(t_test)
#
#     if torch.is_tensor(W_test):
#         W_test = W_test.data.cpu().numpy()
#     else:
#         W_test = np.array(W_test)
#
#     if torch.is_tensor(X_pred):
#         X_pred = X_pred.data.cpu().numpy()
#     else:
#         X_pred = np.array(X_pred)
#
#     if torch.is_tensor(Y_pred):
#         Y_pred = Y_pred.data.cpu().numpy()
#     else:
#         Y_pred = np.array(Y_pred)
#
#     if torch.is_tensor(Y_test):
#         Y_test = Y_test.data.cpu().numpy()
#     else:
#         Y_test = np.array(Y_test)
#
#     if torch.is_tensor(graph):
#         graph = graph.data.cpu().numpy()
#     else:
#         graph = np.array(graph)

    torch.save(model, "BSB"+ model.mode + "-" + model.activation +"20k")
    np.save("BSB"+ model.mode + "-" + model.activation +"20k" + 't_test.npy', t_test)
    np.save("BSB"+ model.mode + "-" + model.activation +"20k" + 'W_test.npy', W_test)
    np.save("BSB"+ model.mode + "-" + model.activation +"20k" + 'X_pred.npy', X_pred)
    np.save("BSB"+ model.mode + "-" + model.activation +"20k" + 'Y_pred.npy', Y_pred)
    np.save("BSB"+ model.mode + "-" + model.activation +"20k" + 'Y_test.npy', Y_test)
    np.save("BSB"+ model.mode + "-" + model.activation +"20k" + 'graph.npy', graph)

    plt.figure()
    plt.plot(graph[0], graph[1])
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.yscale("log")
    plt.title('Evolution of the training loss')
    plt.savefig("TrainLoss20k")

    samples = 5
    plt.figure()
    # plt.plot(t_test[0:1, :, 0].T, Y_pred[0:1, :, 0].T, 'b', label='Learned $u(t,X_t)$')
    # plt.plot(t_test[0:1, :, 0].T, Y_test[0:1, :, 0].T, 'r--', label='Exact $u(t,X_t)$')
    # plt.plot(t_test[0:1, -1, 0], Y_test[0:1, -1, 0], 'ko', label='$Y_T = u(T,X_T)$')
    plt.plot(t_test[1:samples, :, 0].T, Y_pred[1:samples, :, 0].T, 'b')
    plt.plot(t_test[1:samples, :, 0].T, Y_test[1:samples, :, 0].T, 'r--')
    plt.plot(t_test[1:samples, -1, 0], Y_test[1:samples, -1, 0], 'ko')
    #
    plt.plot([0], Y_test[0, 0, 0], 'ks', label='$Y_0 = u(0,X_0)$')
    #
    print(f"Y_test[0, 0, 0] {Y_test[0, 0, 0]}")
    # plt.xlabel('$t$')
    # plt.ylabel('$Y_t = u(t,X_t)$')
    # plt.title(str(D) + '-dimensional Black-Scholes-Barenblatt, ' + model.mode + "-" + model.activation)
    plt.legend()
    plt.savefig("path20k")
    plt.show()


    errors = np.sqrt((Y_test - Y_pred) ** 2 / Y_test ** 2)
    mean_errors = np.mean(errors, 0)
    std_errors = np.std(errors, 0)

    plt.figure()
    plt.plot(t_test[0, :, 0], mean_errors, 'b', label='mean')
    plt.plot(t_test[0, :, 0], mean_errors + 2 * std_errors, 'r--', label='mean + two standard deviations')
    plt.xlabel('$t$')
    plt.ylabel('relative error')
    plt.title(str(D) + '-dimensional-Black-Scholes-Barenblatt-' + model.mode + "-" + model.activation)
    plt.legend()
    plt.savefig(str(D) + '-dimensional-Black-Scholes-Barenblatt-' + model.mode + "-" + model.activation +"20k")
    plt.show()
    cwd = os.getcwd()
    print(cwd)

    text_file = open("Output.txt", "w")
    text_file.write(f"where is this file\nhere: {cwd}")
    text_file.close()

if __name__ == "__main__":
    tot = time.time()
    M = 100  # number of trajectories (batch size)
    N = 50  # number of time snapshots
    D = 100  # number of dimensions

    layers = [D + 1] + 4 * [256] + [1]

    Xi = np.array([1.0, 0.5] * int(D / 2))[None, :]
    T = 1.0

    "Available architectures"
    mode = "FC"  # FC, Resnet and NAIS-Net are available
    activation = "Sine" #"ReLU"  # Sine and ReLU are available
    model = BlackScholesBarenblatt(Xi, T,
                                   M, N, D,
                                   layers, mode, activation)
    # run_model(model, 2*10**3, 1e-3)
    run_model(model, 2 * 10 ** 4, 1e-3)
    # run_model(model, 2 * 10, 1e-3)