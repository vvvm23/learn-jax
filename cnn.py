import jax.numpy as np
import jax.lax as lax
from jax import grad, jit, vmap, value_and_grad
from jax import random

from jax.experimental import optimizers

import torch
from torchvision import datasets, transforms

from functools import partial

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
            datasets.MNIST('../data', train=False, 
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])),
            batch_size=batch_size, shuffle=True, num_workers=4)
    elif name == 'fashion':
        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('../data', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])),
            batch_size=batch_size, shuffle=True, num_workers=4)

        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('../data', train=False, 
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])),
            batch_size=batch_size, shuffle=True, num_workers=4)
    else:
        print("> Unrecognised dataset name! Exiting!")
        exit()

    return train_loader, test_loader

def init_conv(
        key,
        in_channels: int, 
        out_channels: int,
        kernel_size: int,
        stride: int,
        scale: float = 1e-2,
    ):
    wkey, bkey = random.split(key)
    param = {
        'w': scale * random.normal(wkey, (out_channels, in_channels, kernel_size, kernel_size)),
        'b': scale * random.normal(bkey, (out_channels, 1, 1))
    }
    forward_fn = partial(lax.conv, window_strides=(stride, stride), padding='SAME')
    return param, forward_fn

def init_net(key, channels, kernels, strides, fc_dim, nb_classes, scale=1e-2):
    wkey, bkey, *keys = random.split(key, len(kernels)+2)
    conv =  [init_conv(k, n, m, sz, st, scale=scale) for k, n, m, sz, st in zip(keys, channels[:-1], channels[1:], kernels, strides)]
    
    fc = {
        'w': scale * random.normal(wkey, (nb_classes, fc_dim)),
        'b': scale * random.normal(bkey, (nb_classes,))
    }

    return conv, fc

def relu(x):
    return np.maximum(0, x)

def net_forward(conv, fc, x):
    N = x.shape[0]
    for p, fn in conv:
        x = relu(fn(x, p['w']) + p['b'])
    x = x.reshape(N, fc['w'].shape[-1])
    x = vmap(np.dot, in_axes=(None, 0), out_axes=0)(fc['w'], x) + fc['b']
    return x

def one_hot(x, nb_classes, dtype=np.float32):
    return jnp.array(x[:, None] == jnp.arange(nb_classes), dtype=dtype)

def loss_fn(conv, fc, batch_x, batch_y):
    N = batch_x.shape[0]
    logits = net_forward(conv, fc, batch_x)

    predicted_class = np.argmax(logits, axis=-1)
    true_class = np.argmax(batch_y, axis=-1)
    nb_correct = np.sum(predicted_class == true_class)

    return -np.sum(logits * batch_y) / N, nb_correct / N

@jit
def update(conv, fc, x, y, opt, opt_state):
    _, opt_update, get_params = opt
    (loss, acc), grad = value_and_grad(loss_fn, has_aux=True)(conv, fc, x, y)
    opt_state = opt_update(0, grad, opt_state)
    new_params = get_params(opt_state)

    fc = new_params['fc']
    new_conv = new_params['conv']
    conv = [nc, oc[1] for oc, nc in zip(conv, new_conv)]
    return conv, fc, opt_state, loss, acc

def train():
    pass

if __name__ == '__main__':
    key = random.PRNGKey(123)
    conv_key, key = random.split(key)
    param, fn = init_conv(conv_key, 3, 16, 3, stride=1)

    conv, fc = init_net(key, [3, 16, 32, 32], [3, 3, 3], [2, 1, 2], 7*7*32, 10)
    train_loader, test_loader = setup_loaders(batch_size = 128)
    opt_init, opt_update, get_params = optimizers.adam(1e-3)
    opt_state = opt_init({'conv': [c[0] for c in conv], 'fc': fc})

    logits = net_forward(conv, fc, X)
    print(logits.shape)

    print(update(conv, fc, ))
