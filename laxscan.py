import jax.numpy as jnp
import jax.lax as lax
from jax import make_jaxpr

# lax.scan
def func1(arr, extra):
    ones = jnp.ones(arr.shape)
    # twos = 2*ones
    def body(carry, aelems):
        ae1, ae2 = aelems
        return (carry + ae1 * ae2 + extra, carry)
    return lax.scan(body, 0., (arr, ones))
    # return lax.scan(body, 0., (arr, twos))
# make_jaxpr(func11)(jnp.arange(16), 5.)
print("func1")
print(func1(jnp.arange(16), 5.))

def func2(arr, extra):
    ones = jnp.ones(arr.shape)
    def body(carry, aelems):
        ae1, ae2 = aelems
        return (carry + ae1 * ae2 + extra, carry)

    def scan(f, init, xs, length=None):
      if xs is None:
        xs = [None] * length
      carry = init
      ys = []
      for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
      return carry, jnp.stack(ys)
    # return scan(body, 0., (arr, ones))
    return scan(body, 0., jnp.stack((arr, ones),1))
print("func2")
print(func2(jnp.arange(16), 5.))

def func3(input, X0, extra):
    def body(carry, aelems):
        carry1, carry2 = carry
        ae1, ae2 = aelems
        carry1 = carry1 + ae1 * ae2 + extra
        carry2 = carry2 + ae1 * ae2 + extra
        return ((carry1, carry2), (carry, 2*carry, carry))
    return lax.scan(body, X0, input)

print("func3")
arr = jnp.arange(16)
ones = jnp.ones(arr.shape)
input = jnp.stack((arr,ones), 1)
X0 = (0., 0.)
print(func3(input, X0,5.))