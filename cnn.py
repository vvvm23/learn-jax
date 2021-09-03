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
        scale: float = 1e-2,
    ):
    wkey, bkey = random.split(key)
    param = {
        'w': scale * random.normal(wkey, (out_channels, in_channels, kernel_size, kernel_size)),
        'b': scale * random.normal(bkey, (out_channels, 1, 1))
    }
    # forward_fn = partial(lax.conv, window_strides=(stride, stride), padding='SAME')
    return param

def init_net(key, channels, kernels, fc_dim, nb_classes, scale=1e-2):
    wkey, bkey, *keys = random.split(key, len(kernels)+2)
    conv =  [init_conv(k, n, m, s, scale=scale) for k, n, m, s in zip(keys, channels[:-1], channels[1:], kernels)]
    
    fc = {
        'w': scale * random.normal(wkey, (nb_classes, fc_dim)),
        'b': scale * random.normal(bkey, (nb_classes,))
    }

    return conv, fc

def relu(x):
    return np.maximum(0, x)

def net_forward(conv, fc, strides, x):
    N = x.shape[0]
    for p, s in zip(conv, strides):
        # x = relu(fn(x, p['w']) + p['b'])
        x = lax.conv(x, p['w'], window_strides=(s, s), padding='SAME')
        x = relu(x + p['b'])
    x = x.reshape(N, fc['w'].shape[-1])
    x = vmap(np.dot, in_axes=(None, 0), out_axes=0)(fc['w'], x) + fc['b']
    return x

def one_hot(x, nb_classes, dtype=np.float32):
    return np.array(x[:, None] == np.arange(nb_classes), dtype=dtype)

def loss_fn(conv, fc, strides, batch_x, batch_y):
    N = batch_x.shape[0]
    logits = net_forward(conv, fc, strides, batch_x)

    predicted_class = np.argmax(logits, axis=-1)
    true_class = np.argmax(batch_y, axis=-1)
    nb_correct = np.sum(predicted_class == true_class)

    return -np.sum(logits * batch_y) / N, nb_correct / N

# @jit
def update(conv, fc, strides, x, y, opt, opt_state):
    _, opt_update, get_params = opt
    (loss, acc), grad = value_and_grad(loss_fn, argnums=[0, 1], has_aux=True)(conv, fc, strides, x, y)
    opt_state = opt_update(0, grad, opt_state)
    conv, fc = get_params(opt_state)

    return conv, fc, opt_state, loss, acc

def train(conv, fc, strides, opt, opt_state, nb_epochs=10):
    train_loader, test_loader = setup_loaders(batch_size = 128)
    for eid in range(nb_epochs):
        train_loss, train_accuracy = 0.0, 0.0
        for bid, (x, y) in enumerate(train_loader):
            N, _, h, w = x.shape
            x = np.array(x)
            y = one_hot(np.array(y), 10)
            conv, fc, opt_state, loss, acc = update(conv, fc, strides, x, y, opt, opt_state)
            train_loss += loss
            train_accuracy += acc

        test_loss, test_accuracy = 0.0, 0.0
        for bid, (x, y) in enumerate(test_loader):
            N, _, h, w = x.shape
            x = np.array(x)
            y = one_hot(np.array(y), 10)
            loss, acc = loss_fn(conv, fc, strides, x, y)
            test_loss += loss
            test_accuracy += acc

        msg = (
            f"epoch {eid+1}/{nb_epochs}\n"
            f"      train_loss: {train_loss / len(train_loader)}, train_accuracy: {100.0 * train_accuracy / len(train_loader):.2f}%\n"
            f"      test_loss: {test_loss / len(test_loader)}, test_accuracy: {100.0 * test_accuracy / len(test_loader):.2f}%\n"
        )
        print(msg)

if __name__ == '__main__':
    key = random.PRNGKey(123)
    conv_key, key = random.split(key)
    # param, fn = init_conv(conv_key, 3, 16, 3, stride=1)

    conv, fc = init_net(key, [1, 16, 32, 32], [3, 3, 3], 7*7*32, 10)
    opt_init, opt_update, get_params = optimizers.adam(1e-3)
    opt_state = opt_init((conv, fc))

    train(conv, fc, [2,1,2], (opt_init, opt_update, get_params), opt_state)
    # logits = net_forward(conv, fc, X)
    # print(logits.shape)

    # print(update(conv, fc, ))
