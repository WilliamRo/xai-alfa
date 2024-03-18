from tframe import Classifier
from tframe import mu

from sie_core import th



def get_container():
  model = Classifier(mark=th.mark)
  model.add(mu.Input(sample_shape=th.input_shape))
  return model



def finalize(model):
  assert isinstance(model, Classifier)

  # assert th.num_classes == 2
  model.add(mu.Dense(th.num_classes))
  model.add(mu.Activation('softmax'))

  # Build model
  model.build(batch_metric=['accuracy'], metric=['f1', 'accuracy'])
  return model
