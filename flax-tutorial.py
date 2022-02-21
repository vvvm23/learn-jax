import jax
import jax.numpy as jnp

import optax
import flax.linen as nn

import numpy as np
from typing import Sequence

batch_size = 16
nb_steps = 5_000

real_train_data = np.random.randint(0, 255, (nb_steps, batch_size, 1))
train_data = np.unpackbits(real_train_data.astype(np.uint8), axis=-1)
train_labels = jax.nn.one_hot(real_train_data % 2, 2).astype(jnp.float32).reshape(nb_steps, batch_size, 2)

class MLP(nn.Module):
    features: Sequence[int]

    # define network logic with a single forward pass annotated with @nn.compact
    # alt. use setup function (like pt __init__)
    @nn.compact 
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.relu(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        return x

net = MLP([16, 2])

def loss(params: optax.Params, batch: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    logits = net.apply(params, batch)
    loss_val = optax.sigmoid_binary_cross_entropy(logits, labels).sum(axis=-1)
    return loss_val.mean(), (logits.argmax(axis=-1) == labels.argmax(axis=-1)).sum(axis=-1) / batch_size

def fit(params: optax.Params, opt: optax.GradientTransformation) -> optax.Params:
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state, batch, labels):
        (loss_val, accuracy), grads = jax.value_and_grad(loss, has_aux=True)(params, batch, labels)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val, accuracy
    
    for i, (batch, labels) in enumerate(zip(train_data, train_labels)):
        params, opt_state, loss_val, accuracy = step(params, opt_state, batch, labels)
        if i % 100 == 0:
            print(f"step {i}/{nb_steps} | loss: {loss_val:.5f} | accuracy: {accuracy*100:.2f}%")

    return params

params = net.init(jax.random.PRNGKey(0), train_data[0])
schedule = optax.warmup_cosine_decay_schedule(
  init_value=0.0,
  peak_value=1.0,
  warmup_steps=50,
  decay_steps=5_000,
  end_value=0.0,
)
opt = optax.chain(
    optax.clip(1.0),
    # optax.adamw(learning_rate=1e-3)
    optax.adamw(learning_rate=schedule)
)

params = fit(params, opt)
