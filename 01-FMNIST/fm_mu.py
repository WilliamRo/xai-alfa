from tframe import Classifier
from tframe import mu

from fm_core import th



def get_container(flatten=False):
  model = Classifier(mark=th.mark)
  model.add(mu.Input(sample_shape=[28, 28, 1]))
  if th.centralize_data: model.add(mu.Normalize(mu=th.data_mean))
  if flatten: model.add(mu.Flatten())
  return model


def finalize(model):
  assert isinstance(model, Classifier)
  model.add(mu.Dense(10, activation='softmax'))

  # Build model
  model.build(batch_metric=['accuracy'])
  return model
