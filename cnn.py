import jax.numpy as np
import jax.lax as lax
from jax import grad, jit, vmap, value_and_grad
from jax import random

from jax.experimental import optimizers

import torch
from torchvision import datasets, transforms

from functools import partial

def setup_loaders(batch_size: int, dataset_name: str = 'mnist'):
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

if __name__ == '__main__':
    key = random.PRNGKey(123)
    conv_key, key = random.split(key)
    param, fn = init_conv(conv_key, 3, 16, 3, stride=1)

    xkey, key = random.split(key)
    X = random.normal(xkey, (8, 3, 28, 28))

    conv, fc = init_net(key, [3, 16, 32, 32], [3, 3, 3], [2, 1, 2], 7*7*32, 10)
    logits = net_forward(conv, fc, X)
    print(logits.shape)
