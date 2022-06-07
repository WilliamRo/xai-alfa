from tframe import mu

from dsc_core import th



def get_container():
  net_type = mu.Recurrent if th.use_rnn else mu.Feedforward
  model = mu.Classifier(mark=th.mark, net_type=net_type)
  model.add(mu.Input(sample_shape=th.input_shape))
  return model


def finalize(model, flatten=False):
  assert isinstance(model, mu.Classifier)

  if flatten: model.add(mu.Flatten())
  model.add(mu.Dense(th.output_dim))
  model.add(mu.Activation('softmax'))

  # Build model
  model.build(batch_metric=['accuracy'], last_only=th.use_rnn)
  return model
