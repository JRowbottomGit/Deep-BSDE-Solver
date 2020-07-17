import numpy as onp
import jax
import jax.numpy as jnp
from jax.ops import index, index_add, index_update
from jax import grad, jit, vmap
from functools import partial
import flax
from abc import ABC, abstractmethod
import time

from Models import Resnet, Sine

#Flax FX NN
# Dense layers
class Dense(flax.nn.Module):
  """A learned linear transformation."""
  # 1 A Module is created by defining a subclass of flax.nn.Module
  # and implementing the apply method.
  def apply(self, x, in_features, out_features,
            # init functions are of the form (PrngKey, shape) => init_value
            kernel_init=jax.nn.initializers.lecun_normal(),
            bias_init=jax.nn.initializers.zeros):
    """The main entry point to a Module. Represents the function that
    given inputs and hyper-parameters computes an output. The actual parameters
    (inputs and parameters) are user-controlled, and depend on the actual Module functionality.
    For this example:
      * `x`: the input, an array of shape `(in_features)`.
      * `features`: the number of outputs, an integer.
      * `kernel_init`: the initializer for the kernel.
      * `bias_init`: the initializer for the biases.
    """
    kernel_shape = (in_features, out_features)
    # 2 parameters are declared using self.param(name, shape, init_func) and return an initialized parameter value.
    kernel = self.param('kernel', kernel_shape, kernel_init)
    bias = self.param('bias', (out_features,), bias_init)
    return jnp.dot(x, kernel) + bias

class MLP(flax.nn.Module):
  """Multi Layer Perceptron."""
  def apply(self, x, layers=[101, 256, 256, 256, 256, 1],
            activation_fn=flax.nn.relu):
    print(f"in x {x.shape}")

    # x = jnp.reshape(x,(-1,D+1)
    for i in range(len(layers) - 1):
        x = Dense(x, in_features=layers[i],out_features=layers[i+1])
        x = activation_fn(x)

    print(f"out x {x.shape}")
    return x

class FBSNN(ABC):
    def __init__(self, X0, T, M, N, D, layers, mode, activation):

        self.X0 = jnp.array(X0) #<- array
        self.T = T  # terminal time
        self.M = M  # number of trajectories
        self.N = N  # number of time snapshots
        self.D = D  # number of dimensions
        self.layers = layers
        self.mode = mode  # architecture: FC, Resnet and NAIS-Net are available
        self.activation = activation

        # initialize Flax NN
        if self.mode == "FC":
            _, initial_params = MLP.init_by_shape(
                jax.random.PRNGKey(0),                      #don't forget to randomise this
                [((1, D+1), jnp.float32)])
            self.nnet = flax.nn.Model(MLP, initial_params)

        # Record the loss
        self.training_loss = []
        self.iteration = []

    def net_u(self, model, t, X):  # M x 1, M x D
        # input = torch.cat((t, X), 1)   # M x D+1
        input = jnp.concatenate((t, X), 1)  # M x D+1
        print(f"net U input shape {input.shape}")
        u = model(input)  # M x 1  <-should work for both torch and flax
        u_scalar = jnp.reshape(u, ())
        # Du = torch.autograd.grad(outputs=[u], inputs=[X], grad_outputs=torch.ones_like(u), allow_unused=True,
        #                          retain_graph=True, create_graph=True)[0]
        print(f"net U u shape {u.shape}")
        gradu = jax.grad(model, argnums=0)  ##do arg nums
        Du = gradu(input)
        return u, Du

    # def Dg_tf(self, X):  # M x D
    #
    #     g = self.g_tf(X)
    #     Dg = torch.autograd.grad(outputs=[g], inputs=[X], grad_outputs=torch.ones_like(g), allow_unused=True,
    #                              retain_graph=True, create_graph=True)[0]  # M x D
    #     return Dg

#broadcast over the M dimension in t and W - do this later..
    # @jax.vmap  #https://jax.readthedocs.io/en/latest/jax.html?highlight=vmap#jax.vmap
    def loss_function(self, model, t, W, Xzero):
        loss = 0
        X_list = []
        Y_list = []
        t0 = t[:, 0, :]
        W0 = W[:, 0, :]
        # X0 = Xzero.repeat(self.M, 1).view(self.M, self.D)  # M x D repeat numpy function
        X0 = Xzero.repeat(self.M, 1).reshape((self.M, self.D))  # M x D repeat numpy function
        Y0, Z0 = self.net_u(model, t0, X0)  # M x 1, M x D
        X_list.append(X0)
        Y_list.append(Y0)

        for n in range(0, self.N):
            t1 = t[:, n + 1, :]
            W1 = W[:, n + 1, :]
            # X1 = X0 + self.mu_tf(t0, X0, Y0, Z0) * (t1 - t0) + torch.squeeze(
            #     torch.matmul(self.sigma_tf(t0, X0, Y0), (W1 - W0).unsqueeze(-1)), dim=-1)

            X1 = X0 + self.mu_tf(t0, X0, Y0, Z0) * (t1 - t0) + jnp.dot(self.sigma_tf(t0, X0, Y0),(W1 - W0))

            # Y1_tilde = Y0 + self.phi_tf(t0, X0, Y0, Z0) * (t1 - t0) + torch.sum(
            #     Z0 * torch.squeeze(torch.matmul(self.sigma_tf(t0, X0, Y0), (W1 - W0).unsqueeze(-1))), dim=1,
            #     keepdim=True)

            Y1_tilde = Y0 + self.phi_tf(t0, X0, Y0, Z0) * (t1 - t0) +  jnp.dot(jnp.dot(Z0.T, self.sigma_tf(t0, X0, Y0)),(W1 - W0))

            Y1, Z1 = self.net_u(model, t1, X1)

            loss += jnp.sum(jnp.power(Y1 - Y1_tilde, 2))

            t0 = t1
            W0 = W1
            X0 = X1
            Y0 = Y1
            Z0 = Z1

            X_list.append(X0)
            Y_list.append(Y0)

        loss += jnp.sum(jnp.power(Y1 - self.g_tf(X1), 2))
        # loss += torch.sum(torch.pow(Z1 - self.Dg_tf(X1), 2))

        X = jnp.stack(X_list, dim=1)
        Y = jnp.stack(Y_list, dim=1)

        return loss, X, Y, Y[0, 0, 0]

    def fetch_minibatch(self):  # Generate time + a Brownian motion
        T = self.T
        M = self.M
        N = self.N
        D = self.D

        Dt = jnp.zeros((M, N + 1, 1))  # M x (N+1) x 1
        DW = jnp.zeros((M, N + 1, D))  # M x (N+1) x D

        dt = T / N

        #Dt[:, 1:, :] = dt
        new_Dt = index_update(Dt, index[:, 1:, :], 1.)

        #DW[:, 1:, :] = jnp.sqrt(dt) * jnp.random.normal(size=(M, N, D))
        new_DW = index_update(DW, index[:, 1:, :], 1.)

        t = jnp.cumsum(new_Dt, axis=1)  # M x (N+1) x 1
        W = jnp.cumsum(new_DW, axis=1)  # M x (N+1) x D
        # t = torch.from_numpy(t).float().to(self.device) <- cancel these out so stays as numpy
        # W = torch.from_numpy(W).float().to(self.device) <- cancel these out so stays as numpy

        return t, W

    # # Compute loss and accuracy. We use jnp (jax.numpy) which can run on device (GPU or TPU).
    # def compute_metrics(logits, labels):
    #     loss = jnp.mean(cross_entropy_loss(logits, labels))
    #     accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    #     return {'loss': loss, 'accuracy': accuracy}

    # jax.jit traces the train_step function and compiles into fused device operations that run on GPU or TPU.
    # @jax.jit
    @partial(jit, static_argnums=(0,)) #https://github.com/google/jax/issues/1251
    def train_step(self, optimizer, t_batch, W_batch, X0):
    # def train_step(optimizer, batch):
    #need to bring self.loss_function into the body of train_step to generate all the values
    #or... model is the first argument, can pass the other things as arguments but when calcing grads only do wrt model/thetas
        def loss_fn(model, t_batch, W_batch, X0):   #<-this needs to be only a function of model
            loss, X_pred, Y_pred, Y0_pred = self.loss_function(model, t_batch, W_batch, X0)
            return loss, X_pred, Y_pred, Y0_pred
        grad = jax.grad(loss_fn,argnums=(0))(optimizer.target, t_batch, W_batch, X0)  #fix argnums to just 0?? and (model, t_batch, W_batch, X0)
        optimizer = optimizer.apply_gradient(grad) #https://flax.readthedocs.io/en/latest/flax.optim.html
        return optimizer

    # # Making model predictions is as simple as calling model(input):
    # @jax.jit
    # def eval(model, eval_ds):
    #     logits = model(eval_ds['image'] / 255.0)
    #     return compute_metrics(logits, eval_ds['label'])

    def train(self, N_Iter, learning_rate):
        loss_temp = jnp.array([])

        previous_it = 0
        if self.iteration != []:
            previous_it = self.iteration[-1]

        # Optimizers
        # self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # _, initial_params = MLP.init_by_shape(
        #     jax.random.PRNGKey(0),                      #don't forget to randomise this
        #     [((1, self.D+1), jnp.float32)])
        # nnet = flax.nn.Model(MLP, initial_params)

        optimizer = flax.optim.Momentum(learning_rate=0.1, beta=0.9).create(self.nnet) #<- this was a problem being an attridute
        # optimizer = flax.optim.Momentum(learning_rate=0.1, beta=0.9).create(nnet)

        start_time = time.time()
        for it in range(previous_it, previous_it + N_Iter):  #<-the 2*10**4 epochs

            # self.optimizer.zero_grad()
            t_batch, W_batch = self.fetch_minibatch()  # M x (N+1) x 1, M x (N+1) x D
            X0 = self.X0
            optimizer = self.train_step(optimizer, t_batch, W_batch, X0)


            # SORT AFTER TRAINING RUNS WITH NO ERRORS
            # loss_temp = np.append(loss_temp, loss.cpu().detach().numpy())
            #
            # # Print
            # if it % 100 == 0:
            #     elapsed = time.time() - start_time
            #     print('It: %d, Loss: %.3e, Y0: %.3f, Time: %.2f, Learning Rate: %.3e' %
            #           (it, loss, Y0_pred, elapsed, learning_rate))
            #     start_time = time.time()
            #
            # # Loss
            # if it % 100 == 0:
            #     self.training_loss.append(loss_temp.mean())
            #     loss_temp = np.array([])
            #
            #     self.iteration.append(it)
            #
            # graph = np.stack((self.iteration, self.training_loss))
        # return graph

    def predict(self, Xi_star, t_star, W_star):
        Xi_star = torch.from_numpy(Xi_star).float().to(self.device)
        Xi_star.requires_grad = True
        loss, X_star, Y_star, Y0_pred = self.loss_function(t_star, W_star, Xi_star)

        return X_star, Y_star

    ###########################################################################
    ############################# Change Here! ################################
    ###########################################################################
    #abstract pde coefficients for each example
    @abstractmethod
    def phi_tf(self, t, X, Y, Z):  # M x 1, M x D, M x 1, M x D
        pass  # M x1

    @abstractmethod
    def g_tf(self, X):  # M x D
        pass  # M x 1

    @abstractmethod
    def mu_tf(self, t, X, Y, Z):  # M x 1, M x D, M x 1, M x D
        M = self.M
        D = self.D
        return torch.zeros([M, D]).to(self.device)  # M x D

    @abstractmethod
    def sigma_tf(self, t, X, Y):  # M x 1, M x D, M x 1
        M = self.M
        D = self.D
        return torch.diag_embed(torch.ones([M, D])).to(self.device)  # M x D x D
    ###########################################################################