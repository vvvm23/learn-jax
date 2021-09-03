import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
from jax import random

from jax.scipy.special import logsumexp
from jax.experimental import optimizers

import torch
from torchvision import datasets, transforms

def setup_loaders(batch_size: int, name: str = 'mnist'):
    if name == 'mnist':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batch_size, shuffle=True, num_workers=4)

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batch_size, shuffle=True, num_workers=4)
    elif name == 'fashion':
        pass
    else:
        print("> Unrecognised dataset name! Exiting!")
        exit()

    return train_loader, test_loader

def init_net(sizes, key):
    keys = random.split(key, len(sizes))

    def init_linear(m, n, key, scale=1e-2):
        w_key, b_key = random.split(key)
        return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n, ))
    return [init_linear(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

@jit
def relu_layer(params, x):
    x = jnp.dot(params[0], x) + params[1]
    return jnp.maximum(0, x)

def net_forward(params, x):
    for w, b in params[:-1]:
        x = relu_layer([w, b], x)
    w_out, b_out = params[-1]
    logits = jnp.dot(w_out, x) + b_out
    return logits - logsumexp(logits)
batch_forward = vmap(net_forward, in_axes=(None, 0), out_axes=0)

def one_hot(x, nb_classes, dtype=jnp.float32):
    return jnp.array(x[:, None] == jnp.arange(nb_classes), dtype=dtype)
one_hot = jit(one_hot, static_argnums=1)

def loss_fn(params, batch_x, batch_y):
    N = batch_x.shape[0]
    pred = batch_forward(params, batch_x)

    predicted_class = jnp.argmax(pred, axis=-1)
    true_class = jnp.argmax(batch_y, axis=-1)
    nb_correct = jnp.sum(predicted_class == true_class)

    return -jnp.sum(pred * batch_y) / N, nb_correct / N

@jit
def update(params, x, y, opt_state):
    (loss, acc), grad = value_and_grad(loss_fn, has_aux=True)(params, x, y)
    opt_state = opt_update(0, grad, opt_state)
    return get_params(opt_state), opt_state, loss, acc

def train(opt_state, nb_epochs=10):
    params = get_params(opt_state)

    for eid in range(nb_epochs):
        train_loss, train_accuracy = 0.0, 0.0
        for bid, (x, y) in enumerate(train_loader):
            N, _, h, w = x.shape
            x = jnp.array(x).reshape(N, w*h)
            y = one_hot(jnp.array(y), 10)
            params, opt_state, loss, acc = update(params, x, y, opt_state)
            train_loss += loss
            train_accuracy += acc

        test_loss, test_accuracy = 0.0, 0.0
        for bid, (x, y) in enumerate(test_loader):
            N, _, h, w = x.shape
            x = jnp.array(x).reshape(N, w*h)
            y = one_hot(jnp.array(y), 10)
            loss, acc = loss_fn(params, x, y)
            test_loss += loss
            test_accuracy += acc

        msg = (
            f"epoch {eid+1}/{nb_epochs}\n"
            f"      train_loss: {train_loss / len(train_loader)}, train_accuracy: {100.0 * train_accuracy / len(train_loader):.2f}%\n"
            f"      test_loss: {test_loss / len(test_loader)}, test_accuracy: {100.0 * test_accuracy / len(test_loader):.2f}%\n"
        )
        print(msg)

if __name__ == '__main__':
    key = random.PRNGKey(777)
    train_loader, test_loader = setup_loaders(batch_size = 128)
    params = init_net([28*28, 512, 256, 10], key)

    step_size = 1e-3
    opt_init, opt_update, get_params = optimizers.adam(step_size)
    opt_state = opt_init(params)

    train(opt_state, nb_epochs=20)
