from mg_core import th
from tframe import mu, pedia



def get_container(flatten_D_input=False) -> mu.GAN:
  gan = mu.GAN(
    mark=th.mark, G_input_shape=[th.z_dim], D_input_shape=th.input_shape)

  if flatten_D_input: gan.D.add(mu.Flatten())

  return gan


def finalize(gan):
  assert isinstance(gan, mu.GAN)

  # Add last layer to discriminator
  gan.D.add(mu.HyperDense(1))
  gan.D.add(mu.Activation('sigmoid', set_logits=True))

  # Build model
  gan.build(loss=pedia.cross_entropy)
  return gan


def vanilla17(gan):
  assert isinstance(gan, mu.GAN)

  # Generator
  gan.G.add(mu.Dense(64))
  if th.use_batchnorm: gan.G.add(mu.BatchNorm())
  gan.G.add(mu.Activation('relu'))

  gan.G.add(mu.Dense(128))
  if th.use_batchnorm: gan.G.add(mu.BatchNorm())
  gan.G.add(mu.Activation('relu'))

  gan.G.add(mu.Dense(128))
  if th.use_batchnorm: gan.G.add(mu.BatchNorm())
  gan.G.add(mu.Activation('relu'))

  gan.G.add(mu.Dense(784))
  gan.G.add(mu.Activation('tanh'))

  gan.G.add(mu.Rescale(from_scale=[-1., 1.], to_scale=[0., 1.]))
  gan.G.add(mu.Reshape(shape=th.input_shape))

  # Discriminator
  gan.D.add(mu.Rescale(from_scale=[0., 1.], to_scale=[-1., 1.]))

  gan.D.add(mu.Dense(128))
  gan.G.add(mu.Activation('lrelu'))

  gan.D.add(mu.Dense(128))
  if th.use_batchnorm: gan.G.add(mu.BatchNorm())
  gan.G.add(mu.Activation('lrelu'))

  gan.D.add(mu.Dense(64))
  if th.use_batchnorm: gan.G.add(mu.BatchNorm())
  gan.G.add(mu.Activation('lrelu'))


def dcgan17(gan):
  assert isinstance(gan, mu.GAN)

  # Generator
  gan.G.add(mu.Dense(128 * 7 * 7))
  gan.G.add(mu.Activation('relu'))
  gan.G.add(mu.Reshape(shape=[7, 7, 128]))

  gan.G.add(mu.Deconv2D(64, 5, 2))
  gan.G.add(mu.Activation('relu'))

  gan.G.add(mu.Deconv2D(1, 5, 2))
  gan.G.add(mu.Activation('tanh'))

  gan.G.add(mu.Rescale(from_scale=[-1., 1.], to_scale=[0., 1.]))

  # Discriminator
  gan.D.add(mu.Rescale(from_scale=[0., 1.], to_scale=[-1., 1.]))

  gan.D.add(mu.Conv2D(64, 5, 2))
  gan.G.add(mu.Activation('lrelu'))

  gan.D.add(mu.Conv2D(128, 5, 2))
  gan.G.add(mu.Activation('lrelu'))

  gan.D.add(mu.Reshape())
