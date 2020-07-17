import time
import numpy.random as npr
from jax.api import jit, grad
from jax.scipy.special import logsumexp
import jax.numpy as jnp
from functools import partial
from examples import datasets

class JAXNN:
  def __init__(self, scale=None, layer_sizes=None, params=None, step_size=None):
    if params is None:
      self.params = self.init_random_params(scale, layer_sizes)
    else:
      self.params = params

    self.step_size = step_size

  def init_random_params(self, scale, layer_sizes, rng=npr.RandomState(0)):
    return [(scale * rng.randn(m, n), scale * rng.randn(n))
            for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])]

  def predict(self, params, inputs):
    activations = inputs
    for w, b in params[:-1]:
      outputs = jnp.dot(activations, w) + b
      activations = jnp.tanh(outputs)

    final_w, final_b = params[-1]
    logits = jnp.dot(activations, final_w) + final_b
    return logits - logsumexp(logits, axis=1, keepdims=True)

  @partial(jit, static_argnums=0)
  def update(self, params, batch):
    grads = grad(self.loss)(params, batch)
    return [(w - self.step_size * dw, b - self.step_size * db)
            for (w, b), (dw, db) in zip(params, grads)]

  def loss(self, params, batch):
    inputs, targets = batch
    preds = self.predict(params, inputs)
    return -jnp.mean(jnp.sum(preds * targets, axis=1))

  def accuracy(self, params, batch):
    inputs, targets = batch
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(self.predict(params, inputs), axis=1)
    return jnp.mean(predicted_class == target_class)

if __name__ == "__main__":
  layer_sizes = [784, 1024, 1024, 10]
  param_scale = 0.1
  step_size = 0.001
  num_epochs = 10
  batch_size = 128

  model = JAXNN(scale=param_scale, layer_sizes=layer_sizes, step_size=step_size)

  train_images, train_labels, test_images, test_labels = datasets.mnist()
  num_train = train_images.shape[0]
  num_complete_batches, leftover = divmod(num_train, batch_size)
  num_batches = num_complete_batches + bool(leftover)

  def data_stream():
    rng = npr.RandomState(0)
    while True:
      perm = rng.permutation(num_train)
      for i in range(num_batches):
        batch_idx = perm[i * batch_size:(i + 1) * batch_size]
        yield train_images[batch_idx], train_labels[batch_idx]
  batches = data_stream()

  # params = init_random_params(param_scale, layer_sizes)

  for epoch in range(num_epochs):
    start_time = time.time()
    for _ in range(num_batches):
      params = model.params
      new_params = model.update(params, next(batches))
      model = JAXNN(params=new_params, step_size=step_size)

    epoch_time = time.time() - start_time

    train_acc = model.accuracy(model.params, (train_images, train_labels))
    test_acc = model.accuracy(model.params, (test_images, test_labels))
    print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
    print("Training set accuracy {}".format(train_acc))
    print("Test set accuracy {}".format(test_acc))