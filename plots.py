import numpy as np
import matplotlib.pyplot as plt

t_test = np.load('t_test.npy')
W_test = np.load('W_test.npy')
t_plot = np.load('t_plot.npy')
X_pred = np.load('X_pred.npy')
Y_pred = np.load('Y_pred.npy')
Y_tilde_pred = np.load('Y_tilde_pred.npy')
Y_test = np.load('Y_test.npy')
graph = np.load('graph.npy')

#Training Loss
plt.figure()
plt.plot(graph[0], graph[1])
plt.xlabel('Iterations')
plt.ylabel('Value')
plt.yscale("log")
plt.title('Evolution of the training loss')
plt.legend()
plt.savefig("Jax_TrainLoss")
plt.show()

# plot X path
plt.figure()
plt.plot(t_plot[0,:,0], X_pred[2,:,0]) #1st dimension of X
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('X path sample')
plt.legend()
plt.savefig("Jax_X_path_sample")
plt.show()

#Y predicted and exact - samples
samples = 4
plt.figure()
for i in range(samples):
    plt.plot(t_plot[i, :, 0], Y_pred[i, :, 0], 'b', label='Learned $u(t,X_t)$')
    plt.plot(t_plot[i, :, 0], Y_test[i, :, 0], 'r--', label='Exact $u(t,X_t)$')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Y values')
plt.legend()
plt.savefig("Jax_Y_pred_and_test")
plt.show()

#Average over test batch
plt.figure()
Y_pred_mean = np.mean(Y_pred, 0)
Y_test_mean = np.mean(Y_test, 0)
plt.plot(t_plot[0, :, 0], Y_pred_mean[:, 0], 'b', label='Learned $u(t,X_t)$')
plt.plot(t_plot[0, :, 0], Y_test_mean[:, 0], 'r--', label='Exact $u(t,X_t)$')
plt.legend()
plt.savefig("Jax_average_Y_pred_and_test")
plt.show()

#mean error
errors = np.sqrt((Y_test - Y_pred) ** 2 / Y_test ** 2)
mean_errors = np.mean(errors, 0)
std_errors = np.std(errors, 0)
plt.figure()
plt.plot(t_plot[0, :, 0], mean_errors, 'b', label='mean')
plt.plot(t_plot[0, :, 0], mean_errors + 2 * std_errors, 'r--', label='mean + two standard deviations')
D=100
plt.xlabel('$t$')
plt.ylabel('relative error')
plt.title('Jax' + str(D) + '-dimensional-Black-Scholes-Barenblatt-' + "FC" + "-" + "ReLu" + "_JRJaxVec")
plt.legend()
plt.savefig('Jax' + str(D) + '-dimensional-Black-Scholes-Barenblatt-' + "FC" + "-" + "ReLu" + "_JRJaxVec")
plt.show()

# cwd = os.getcwd()
# print(cwd)
# text_file = open("JRJaxVec_Output.txt", "w")
# text_file.write(f"where is this file\nhere: {cwd}")
# text_file.close()