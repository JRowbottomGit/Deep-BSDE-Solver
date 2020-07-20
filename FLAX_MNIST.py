# Annotated full end-to-end MNIST example
import jax
import flax
# Load vanilla NumPy for use on host.

import numpy as onp
# JAX has a re-implemented NumPy that runs on GPU and TPU
import jax.numpy as jnp

# Flax can use any data loading pipeline. We use TF datasets.
import tensorflow as tf
import tensorflow_datasets as tfds
# A Flax “module” lets you write a normal function, which defines learnable parameters in-line.
# In this case, we define a simple convolutional neural network.
# Each call to flax.nn.Conv defines a learnable kernel.
class CNN(flax.nn.Module):
  def apply(self, x):
    print(f"in x {x.shape}")
    x = flax.nn.Conv(x, features=32, kernel_size=(3, 3))
    x = flax.nn.relu(x)
    x = flax.nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = flax.nn.Conv(x, features=64, kernel_size=(3, 3))
    x = flax.nn.relu(x)
    x = flax.nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))
    x = flax.nn.Dense(x, features=256)
    x = flax.nn.relu(x)
    x = flax.nn.Dense(x, features=10)
    x = flax.nn.log_softmax(x)
    print(f"out x {x.shape}")
    return x

# Dense layers
class Dense(flax.nn.Module):
  """A learned linear transformation."""
  # 1 A Module is created by defining a subclass of flax.nn.Module
  # and implementing the apply method.
  def apply(self, x, features,
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
    in_features = x.shape[-1]
    kernel_shape = (in_features, features)
    # 2 parameters are declared using self.param(name, shape, init_func) and return an initialized parameter value.
    kernel = self.param('kernel', kernel_shape, kernel_init)
    bias = self.param('bias', (features,), bias_init)
    return jnp.dot(x, kernel) + bias

# Composition of 3 hidden layers + 1 dense output
class MLP(flax.nn.Module):
  """Multi Layer Perceptron."""
  def apply(self, x,
            hidden_features=10,
            output_features=10,
            activation_fn=flax.nn.relu):
    print(f"in x {x.shape}")
    x = jnp.reshape(x,(-1,28*28))
    x = Dense(x, hidden_features)
    x = activation_fn(x)
    x = Dense(x, hidden_features)
    x = activation_fn(x)
    x = Dense(x, hidden_features)
    x = activation_fn(x)
    x = Dense(x, output_features)
    x = flax.nn.log_softmax(x)
    print(f"out x {x.shape}")
    return x

# jax.vmap allows us to define the cross_entropy_loss function as if it acts on a single sample. jax.vmap automatically vectorizes code efficiently to run on entire batches.
@jax.vmap
def cross_entropy_loss(logits, label):
  return -logits[label]

# Compute loss and accuracy. We use jnp (jax.numpy) which can run on device (GPU or TPU).
def compute_metrics(logits, labels):
  loss = jnp.mean(cross_entropy_loss(logits, labels))
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  return {'loss': loss, 'accuracy': accuracy}

# jax.jit traces the train_step function and compiles into fused device operations that run on GPU or TPU.
@jax.jit
def train_step(optimizer, batch):
  def loss_fn(model):
    logits = model(batch['image'])
    loss = jnp.mean(cross_entropy_loss(
        logits, batch['label']))
    return loss
  grad = jax.grad(loss_fn)(optimizer.target)
  optimizer = optimizer.apply_gradient(grad)
  return optimizer

# Making model predictions is as simple as calling model(input):
@jax.jit
def eval(model, eval_ds):
  logits = model(eval_ds['image'] / 255.0)
  return compute_metrics(logits, eval_ds['label'])

# Main train loop
def train():
# Load, convert dtypes, and shuffle MNIST.

  train_ds = tfds.load('mnist', split=tfds.Split.TRAIN)
  train_ds = train_ds.map(lambda x: {'image': tf.cast(x['image'], tf.float32),
                                     'label': tf.cast(x['label'], tf.int32)})
  train_ds = train_ds.cache().shuffle(1000).batch(128)
  test_ds = tfds.as_numpy(tfds.load(
      'mnist', split=tfds.Split.TEST, batch_size=-1))
  test_ds = {'image': test_ds['image'].astype(jnp.float32),
             'label': test_ds['label'].astype(jnp.int32)}

  # Create a new model, running all necessary initializers.
  # The parameters are stored as nested dicts on model.params.

  # _, initial_params = CNN.init_by_shape(
  # jax.random.PRNGKey(0),
  #  [((1, 28, 28, 1), jnp.float32)])
  # model = flax.nn.Model(CNN, initial_params)

  _, initial_params = MLP.init_by_shape(
  jax.random.PRNGKey(0),
  [((1, 28, 28, 1), jnp.float32)])
  model = flax.nn.Model(MLP, initial_params)


  # Define an optimizer. At any particular optimzation step, optimizer.target contains the model.
  optimizer = flax.optim.Momentum(
      learning_rate=0.1, beta=0.9).create(model)

  # Run an optimization step for each batch of training
  for epoch in range(3):
    print(f"epoch {epoch}")
    for batch in tfds.as_numpy(train_ds):
      # print(f"batch {batch}")
      batch['image'] = batch['image'] / 255.0
      optimizer = train_step(optimizer, batch)

    print("here")
  # Once an epoch, evaluate on the test set.
    metrics = eval(optimizer.target, test_ds)

  # metrics are only retrieved from device when needed on host (like in this print statement)
    print('eval epoch: %d, loss: %.4f, accuracy: %.2f'
         % (epoch+1,
          metrics['loss'], metrics['accuracy'] * 100))

if __name__ == "__main__":
  train()