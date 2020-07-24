
if __name__ == "__main__":
    print("here")
    from jax.lib import xla_bridge
    print(xla_bridge.get_backend().platform)
    print("here2")