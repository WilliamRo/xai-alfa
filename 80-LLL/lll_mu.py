from tframe import Classifier
from tframe import mu
from tframe import tf

from lll_core import th

import lll.springs.moses



def get_container(flatten=False):
  model = Classifier(mark=th.mark)
  model.add(mu.Input(sample_shape=th.input_shape))
  if th.centralize_data: model.add(mu.Normalize(mu=th.data_mean))
  if flatten: model.add(mu.Flatten())
  return model


def finalize(model):
  from tframe import context

  assert isinstance(model, Classifier)
  model.add(mu.Dense(th.output_dim))
  model.add(mu.Activation('softmax'))

  # Build model
  # context.customized_loss_f_net = add_customized_loss_f_net
  model.build(batch_metric=['accuracy'])
  return model
  tf.matmul

