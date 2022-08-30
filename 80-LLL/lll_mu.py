from tframe import Classifier
from tframe import mu
from tframe import tf

from lll_core import th



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
  context.customized_loss_f_net = add_customized_loss_f_net
  model.build(batch_metric=['accuracy'])
  return model


# region: Regularizers

def add_customized_loss_f_net(model: Classifier):
  if th.lll_lambda == 0.: return []

  loss_list = {
    'alpha': reg_alpha,
  }[th.reg_code](model)

  return [tf.multiply(th.lll_lambda, tf.add_n(loss_list), name='reg-alpha')]

def reg_alpha(model: Classifier):
  vars = model.var_list
  shadows = model._shadows
  assert len(vars) == len(shadows)

  loss_list = []
  for v in vars:
    s = shadows[v]
    loss_list.append(tf.reduce_mean(tf.square(s - v)))
  return loss_list

# endregion: Regularizers



