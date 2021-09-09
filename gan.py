import jax
import haiku as hk
import optax

import jax.numpy as np
from torchvision import datasets, transforms

def setup_loaders(batch_size: int, name: str = 'mnist'):
    if name == 'mnist':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                ])),
            batch_size=batch_size, shuffle=True, num_workers=4)

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, 
                transform=transforms.Compose([
                    transforms.ToTensor(),
                ])),
            batch_size=batch_size, shuffle=True, num_workers=4)
    elif name == 'fashion':
        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('../data', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                ])),
            batch_size=batch_size, shuffle=True, num_workers=4)

        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('../data', train=False, 
                transform=transforms.Compose([
                    transforms.ToTensor(),
                ])),
            batch_size=batch_size, shuffle=True, num_workers=4)
    else:
        print("> Unrecognised dataset name! Exiting!")
        exit()

    return train_loader, test_loader

class Generator(hk.Module):
    def __init__(self, 
            output_channels = (32, 1),
        ):
        super().__init__()
        self.output_channels = output_channels

    def __call__(self, x):
        N = x.shape[0]
        x = x * 2.0 - 1.0 # TODO: do we want scaling /within/ the module? probably not, so we can pass to Discriminator..

        x = hk.Linear(64*7*7)(x) # TODO: make this size a parameter
        x = np.reshape(x, (N, 64, 7, 7))

        for oc in self.output_channels:
            x = jax.nn.relu(x)
            x = hk.Conv2DTranspose(output_channels=oc, kernel_shape=(5, 5), stride=2, padding='SAME')(x)
        x = np.tanh(x)
        x = (x + 1.0) / 2.0
        return x

class Discriminator(hk.Module):
    def __init__(self, 
            output_channels = (8, 16, 32, 64, 128),
            strides = (2, 1, 2, 1, 2),
        ):
        super().__init__()
        self.output_channels = output_channels
        self.strides = strides

    def __call__(self, x):
        for oc, st in zip(self.output_channels, self.strides):
            x = hk.Conv2D(output_channels = oc, kernel_shape = (5, 5), stride = st, padding = 'SAME')(x)
            x = jax.nn.leaky_relu(x, negative_slope=0.2)
        x = hk.Flatten()(x)
        logits = hk.Linear(2)(x) # could be 1 also
        return logits

def sparse_softmax_cross_entropy(logits, labels):
  one_hot_labels = jax.nn.one_hot(labels, logits.shape[-1])
  return -np.sum(one_hot_labels * jax.nn.log_softmax(logits), axis=-1)
