import tensorflow as tf
from tframe import Classifier

from tframe.layers import Input, Linear, Activation, Rescale
from tframe.nets.rnn_cells.srn import BasicRNNCell
from tframe.nets.rnn_cells.lstms import BasicLSTMCell, OriginalLSTMCell
from tframe.nets.rnn_cells.amu import AMU
from tframe.models import Recurrent

from tframe.configs.config_base import Config
import tframe.metrics as metrics

from tframe.layers.hyper.dense import Dense

import to_core as core


def typical(th, cells):
  assert isinstance(th, Config)
  # Initiate a model
  model = Classifier(mark=th.mark, net_type=Recurrent)
  # Add layers
  model.add(Input(sample_shape=th.input_shape))
  # Add hidden layers
  if not isinstance(cells, (list, tuple)): cells = [cells]
  for cell in cells: model.add(cell)
  # Build model and return
  output_and_build(model, th)
  return model


def output_and_build(model, th):
  assert isinstance(model, Classifier)
  assert isinstance(th, Config)
  # Add output layer
  model.add(Dense(num_neurons=th.output_dim))
  model.add(Activation('softmax'))

  model.build(metric='accuracy', batch_metric='accuracy', last_only=True)
