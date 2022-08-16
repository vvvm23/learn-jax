import jax
import jax.numpy as jnp

import flax.linen as nn
from flax.training import train_state

import optax
import tensorflow_datasets as tfds
import tensorflow as tf

import numpy as np

from typing import Tuple
import tqdm

from utils import save_image

class Encoder(nn.Module):
    z_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=16, kernel_size=(3,3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2,2), strides=(2,2))

        x = nn.Conv(features=32, kernel_size=(3,3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2,2), strides=(2,2))

        x = nn.Conv(features=64, kernel_size=(3,3))(x)
        x = nn.relu(x)

        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(256)(x)
        x = nn.relu(x)

        z = nn.Dense(self.z_dim*2)(x)
        mean, logvar = jnp.split(z, 2, axis=-1)
        return mean, logvar

class Decoder(nn.Module):

    @nn.compact
    def __call__(self, z):
        z = nn.Dense(256)(z)
        z = nn.relu(z)

        z = nn.Dense(64*7*7)(z)
        z = nn.relu(z)

        z = z.reshape((z.shape[0], 64, 7, 7))

        z = nn.Conv(features=64, kernel_size=(3,3))(z)
        z = nn.relu(z)
        z = jax.image.resize(z, (z.shape[0], 14, 14, 64), method='nearest')

        z = nn.Conv(features=32, kernel_size=(3,3))(z)
        z = nn.relu(z)

        z = jax.image.resize(z, (z.shape[0], 28, 28, 32), method='nearest')

        z = nn.Conv(features=16, kernel_size=(3,3))(z)
        z = nn.relu(z)

        y = nn.Conv(features=1, kernel_size=(3,3))(z)

        return y

def reparameterize(rng, mean, logvar):
    std = jnp.exp(0.5 * logvar)
    eps = jax.random.normal(rng, logvar.shape)
    return mean + eps * std

class VAE(nn.Module):
    z_dim: int

    def setup(self):
        self.encoder = Encoder(self.z_dim)
        self.decoder = Decoder()

    def __call__(self, x, z_rng):
        mean, logvar = self.encoder(x)
        z = reparameterize(z_rng, mean, logvar)
        recon = self.decoder(z)
        return recon, mean, logvar

    def generate(self, z):
        return self.decoder(z)

@jax.vmap
def kl_divergence(mean, logvar):
    return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))

def compute_metrics(recon, x, mean, logvar):
    mse_loss = optax.l2_loss(recon, x).mean()
    kl_loss = kl_divergence(mean, logvar).mean()

    return mse_loss, kl_loss, mse_loss + kl_loss

def model():
    return VAE(z_dim=16)

@jax.jit
def train_step(state, batch, z_rng):
    def loss_fn(params):
        recon, mean, logvar = model().apply({'params': params}, batch, z_rng)

        mse_loss = optax.l2_loss(recon, batch).mean()
        kl_loss = kl_divergence(mean, logvar).mean()
        return mse_loss*28*28 + kl_loss, (mse_loss, kl_loss, mse_loss + kl_loss)
    
    grads, metrics = jax.grad(loss_fn, has_aux=True)(state.params)
    return state.apply_gradients(grads=grads), metrics

@jax.jit
def eval(params, batch, z, z_rng):
    def eval_model(vae):
       recon, mean, logvar = vae(batch, z_rng)
       comparison = jnp.concatenate([
           batch[:8],
           recon[:8]
       ])

       samples = vae.generate(z)

       mse_loss, kl_loss, loss = compute_metrics(recon, batch, mean, logvar)
       return mse_loss, kl_loss, loss, comparison, samples

    return nn.apply(eval_model, model())({'params': params})

def prepare_image(x):
    x = tf.cast(x['image'], tf.float16)
    x = x / 255.
    return x

def main(args):
    tf.config.experimental.set_visible_devices([], 'GPU')

    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)

    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    train_ds = ds_builder.as_dataset(split=tfds.Split.TRAIN)
    train_ds = train_ds.map(prepare_image)
    train_ds = train_ds.cache()
    train_ds = train_ds.repeat()
    train_ds = train_ds.shuffle(50000)
    train_ds = train_ds.batch(64)
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
    train_ds = iter(tfds.as_numpy(train_ds))

    test_ds = ds_builder.as_dataset(split=tfds.Split.TEST)
    test_ds = test_ds.map(prepare_image).batch(10000)
    test_ds = np.array(list(test_ds)[0])
    test_ds = jax.device_put(test_ds)

    init_data = jnp.ones((64, 28, 28, 1), jnp.float16)

    state = train_state.TrainState.create(
        apply_fn=model().apply,
        params=model().init(key, init_data, rng)['params'],
        tx=optax.adam(1e-3)
    )

    rng, z_key, eval_rng = jax.random.split(rng, 3)
    z = jax.random.normal(z_key, (64, 16))
    
    for epoch in range(20):
        epoch_metrics = [0.0, 0.0, 0.0]
        for i in tqdm.tqdm(range(50_000 // 64)):
            batch = next(train_ds)
            rng, key = jax.random.split(rng)
            state, batch_metrics = train_step(state, batch, key)
            epoch_metrics = [epoch_metrics[i] + m for i, m in enumerate(batch_metrics)]

        epoch_metrics = [e / (50_000 // 64) for e in epoch_metrics]
        print('train epoch: {}, loss: {:.6f}, MSE: {:.6f}, KLD: {:.6f}'.format(
            epoch + 1, epoch_metrics[-1], epoch_metrics[0], epoch_metrics[1]
        ))

        mse_loss, kl_loss, loss, comparison, sample = eval(state.params, test_ds, z, eval_rng)
        print('eval epoch: {}, loss: {:.6f}, MSE: {:.6f}, KLD: {:.6f}'.format(
            epoch + 1, loss, mse_loss, kl_loss
        ))
        save_image(comparison, f'results/reconstruction_{epoch}.png', nrow=8)
        save_image(sample, f'results/sample_{epoch}.png', nrow=8)

if __name__ == '__main__':
    main(None)
