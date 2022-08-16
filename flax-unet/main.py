import jax
import jax.numpy as jnp

import flax.linen as nn
from flax.training import train_state

import optax
import tensorflow_datasets as tfds
import tensorflow as tf

import numpy as np

from typing import Tuple, Union
from math import sqrt
import tqdm

from utils import save_image

class ConvBlock(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x):
        x = nn.GroupNorm(num_groups=self.channels)(x)
        x = nn.swish(x)
        x = nn.Conv(features=self.channels, kernel_size=(3,3))(x)
        return x

class ResNetBlock(nn.Module):
    channels: int
    scale_skip: bool = True

    @nn.compact
    def __call__(self, x):
        r = nn.Conv(features=self.channels, kernel_size=(1,1))(x)
        if self.scale_skip:
            r = r / sqrt(2)

        x = ConvBlock(self.channels)(x)
        x = ConvBlock(self.channels)(x)
        return x + r

class DBlock(nn.Module):
    channels: int
    stride: int = 1
    nb_res_blocks: int = 2
    res_scale_skip: bool = True
    use_attn: bool = False
    attn_mlp_mult: int = 2
    attn_nb_heads: int = 8

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(
            features=self.channels, 
            kernel_size=(3,3), 
            strides=(self.stride, self.stride)
        )(x)

        for _ in range(self.nb_res_blocks):
            x = ResNetBlock(self.channels, scale_skip=self.res_scale_skip)(x)

        if self.use_attn:
            raise NotImplementedError

        return x

class UBlock(nn.Module):
    channels: int
    out_channels: int
    stride: int = 1
    nb_res_blocks: int = 2
    res_scale_skip: bool = True
    use_attn: bool = False
    attn_mlp_mult: int = 2
    attn_nb_heads: int = 8

    @nn.compact
    def __call__(self, x, r):
        x = x + r

        for _ in range(self.nb_res_blocks):
            x = ResNetBlock(self.channels, scale_skip=self.res_scale_skip)(x)

        if self.use_attn:
            raise NotImplementedError

        x = nn.ConvTranspose(
            features=self.out_channels, 
            kernel_size=(3,3), 
            strides=(self.stride, self.stride)
        )(x)

        return x

class UNet(nn.Module):
    channels: int
    out_channels: int
    strides: Tuple[int]
    channel_mults: Tuple[int]
    nb_res_blocks: Union[int, Tuple[int]]
    res_scale_skip: bool = True

    def setup(self):
        assert len(self.strides) == len(self.channel_mults)

    @nn.compact
    def __call__(self, x):

        nb_res_blocks = self.nb_res_blocks
        if isinstance(nb_res_blocks, int):
            nb_res_blocks = tuple(nb_res_blocks for _ in range(len(self.strides)))

        skips = [] 
        x = nn.Conv(features=self.channels, kernel_size=(3,3))(x)

        for cm, s, nr in zip(self.channel_mults, self.strides, nb_res_blocks):
            x = DBlock(
                channels=cm*self.channels,
                stride=s,
                nb_res_blocks=nr,
                res_scale_skip=self.res_scale_skip,
            )(x)
            skips.append(x)

        x = jnp.zeros_like(x)
        out_channels = self.channel_mults[-2::-1] + (self.channel_mults[0],)
        for r, cm, com, s, nr in zip(skips[::-1], self.channel_mults[::-1], out_channels, self.strides[::-1], nb_res_blocks[::-1]):
            x = UBlock(
                channels=cm*self.channels,
                out_channels=com*self.channels,
                stride=s,
                nb_res_blocks=nr,
                res_scale_skip=self.res_scale_skip,
            )(x, r)

        x = nn.Dense(self.out_channels)(x)
        return x

def model():
    return UNet(
        channels=16, 
        out_channels=3, 
        strides=(2,2,2,1),
        channel_mults=(1,2,4,8),
        nb_res_blocks=(2,3,4,5)
    )

@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        bx, x = batch
        recon = model().apply({'params': params}, bx)

        loss = optax.l2_loss(recon, x).mean()
        return loss
    
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss

@jax.jit
def eval_step(state, batch):
    def loss_fn(params):
        bx, x = batch
        recon = model().apply({'params': params}, bx)

        loss = optax.l2_loss(recon, x).mean()
        return loss
    return loss_fn(state.params)

@jax.jit
def eval_recon(params, batch):
    bx, x = batch
    recon = model().apply({'params': params}, bx)
    return bx, x, recon

def prepare_image(x):
    x = tf.cast(x['image'], tf.float16)
    x = x / 255.
    bx = tf.image.rgb_to_grayscale(x)
    return bx, x

def main(args):
    tf.config.experimental.set_visible_devices([], 'GPU')

    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)

    ds_builder = tfds.builder('stl10')
    ds_builder.download_and_prepare()
    train_ds = ds_builder.as_dataset('train+unlabelled')
    train_ds = train_ds.map(prepare_image)
    train_ds = train_ds.repeat()
    train_ds = train_ds.shuffle(500)
    train_ds = train_ds.batch(64)
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
    train_ds = iter(tfds.as_numpy(train_ds))

    test_ds = ds_builder.as_dataset('test')
    test_ds = test_ds.map(prepare_image)
    test_ds = test_ds.repeat()
    test_ds = test_ds.batch(64)
    test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)
    test_ds = iter(tfds.as_numpy(test_ds))

    init_data = jnp.ones((64, 96, 96, 1), jnp.float16)

    state = train_state.TrainState.create(
        apply_fn=model().apply,
        params=model().init(key, init_data)['params'],
        tx=optax.adam(1e-3)
    )

    for epoch in range(20):
        train_loss = 0.0
        for i in tqdm.trange(50_000 // 64):
            batch = next(train_ds)
            state, loss = train_step(state, batch)
            train_loss += loss
        train_loss /= (50_000 // 64)

        eval_loss = 0.0
        for i in tqdm.trange(10_000 // 64):
            batch = next(test_ds)
            loss = eval_step(state, batch)
            eval_loss += loss
        eval_loss /= (10_000 // 64)

        print(f"Epoch {epoch+1}/20 | train_loss: {train_loss}, eval_loss: {eval_loss}")

        bx, x, recon = eval_recon(state.params, batch)
        # awful line ahead
        comparison = jnp.concatenate([
            np.asarray(tf.image.grayscale_to_rgb(tf.convert_to_tensor(bx[:8]))),
            x[:8],
            recon[:8],
        ])
        save_image(comparison, f'results/reconstruction_{epoch}.png', nrow=8)

if __name__ == '__main__':
    main(None)
