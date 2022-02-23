import jax
import jax.numpy as jnp

import optax
import flax.linen as nn
from flax.training.train_state import TrainState

import torch
from torchvision import datasets, transforms

import numpy as np
from typing import Sequence

batch_size = 64
nb_epochs = 20

image_shape = (32, 32, 3) # HxWxC (channel last format)
nb_classes = 10

train_loader = torch.utils.data.DataLoader(datasets.CIFAR10('data', download=True, train=True, 
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])), batch_size=batch_size, shuffle=True, num_workers=4)

test_loader = torch.utils.data.DataLoader(datasets.CIFAR10('data', download=True, train=False, 
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])), batch_size=batch_size, shuffle=True, num_workers=4)

# simple CNN with relu activations
class CNN(nn.Module):
    conv_features: Sequence[int]
    mlp_features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for feat in self.conv_features:
            x = nn.Conv(feat, kernel_size = (3, 3))(x)
            x = nn.max_pool(x, window_shape = (2, 2))
            x = nn.relu(x)

        x = x.reshape((x.shape[0], -1))
        for feat in self.mlp_features[:-1]:
            x = nn.relu(nn.Dense(feat)(x))
        x = nn.Dense(self.mlp_features[-1])(x)
        return x
net = CNN([16, 32, 64, 128], [256, nb_classes])

# define loss function (just BCE)
def loss(params: optax.Params, batch: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    logits = net.apply(params, batch)
    loss_val = optax.sigmoid_binary_cross_entropy(logits, labels).sum(axis=-1)
    return loss_val.mean(), (logits.argmax(axis=-1) == labels.argmax(axis=-1)).sum(axis=-1) / batch_size

def fit(params: optax.Params, opt: optax.GradientTransformation) -> optax.Params:
    # bundle together everything in the `TrainState class`
    state = TrainState.create(
        apply_fn=net.apply,
        params=params,
        tx=opt,
    )

    # jit compile the step function
    @jax.jit
    def train_step(state, batch, labels):
        batch = jnp.transpose(batch, axes=(0, 2, 3, 1))
        labels = jax.nn.one_hot(labels, nb_classes)
        (loss_val, accuracy), grads = jax.value_and_grad(loss, has_aux=True)(state.params, batch, labels) # return accuracy as aux
        state = state.apply_gradients(grads=grads) # apply gradients to training state (calls other things internally)
        return state, loss_val, accuracy

    @jax.jit
    def eval_step(params, batch, labels):
        batch = jnp.transpose(batch, axes=(0, 2, 3, 1))
        labels = jax.nn.one_hot(labels, nb_classes)
        loss_val, accuracy = loss(params, batch, labels)
        return loss_val, accuracy
    
    for i in range(nb_epochs):
        train_loss, train_accuracy = 0.0, 0.0
        for batch, labels in train_loader:
            batch, labels = jnp.array(batch), jnp.array(labels)
            state, loss_val, accuracy = train_step(state, batch, labels)

            train_loss += loss_val
            train_accuracy += accuracy

        test_loss, test_accuracy = 0.0, 0.0
        for batch, labels in test_loader:
            batch, labels = jnp.array(batch), jnp.array(labels)
            loss_val, accuracy = eval_step(state.params, batch, labels)

            test_loss += loss_val
            test_accuracy += accuracy

        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader)

        test_loss /= len(test_loader)
        test_accuracy /= len(test_loader)

        print(f"epoch {i+1}/{nb_epochs} | train: {train_loss:.5f} [{train_accuracy*100:.2f}%] | eval: {test_loss:.5f} [{test_accuracy*100:.2f}%]")

    return params

params = net.init(jax.random.PRNGKey(0), jnp.ones((1, *image_shape)).astype(jnp.float32))
opt = optax.chain(
    optax.clip(1.0),
    optax.adamw(learning_rate=1e-4)
)
params = fit(params, opt)
