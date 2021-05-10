import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

if __name__ == '__main__':
    key = random.PRNGKey(777)
    size = 10_000 # change to lower value if needed. requires about 10GiB!
    x = random.normal(key, (size,size), dtype=jnp.float16)
    print(jnp.dot(x, x.T).block_until_ready())
