from tframe import Classifier
from tframe import mu

from fm_core import th



def get_container(flatten=False):
  model = Classifier(mark=th.mark)
  model.add(mu.Input(sample_shape=[28, 28, 1]))
  if th.centralize_data: model.add(mu.Normalize(mu=th.data_mean))
  if flatten: model.add(mu.Flatten())
  return model


def finalize(model, fully_conv=False):
  assert isinstance(model, Classifier)

  if fully_conv:
    model.add(mu.Conv2D(filters=10, kernel_size=1, use_bias=False))
    model.add(mu.GlobalAveragePooling2D())
    model.add(mu.Activation('softmax'))
  else:
    model.add(mu.Dense(10, use_bias=False, activation='softmax'))

  # Build model
  model.build(batch_metric=['accuracy'])
  return model
