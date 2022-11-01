from tframe import Classifier
from tframe.layers import Input, Activation
from tframe.layers.hyper.dense import Dense
from tframe import mu



def init_model():
  from slp_core import th

  # Initiate a model
  model = Classifier(mark=th.mark)
  # Add Input layer
  model.add(Input(sample_shape=[3000, th.channel_num]))
  return model


def output_and_build(model):
  # Add output layers
  model.add(mu.Flatten())
  model.add(Dense(num_neurons=5))
  model.add(Activation('softmax'))
  # Build model and return
  model.build(metric='accuracy', batch_metric='accuracy')
  return model


# region: Blocks

def conv1d(kernel_size, filters):
  """Small orange block"""
  return mu.Conv1D(filters, kernel_size, activation='relu')

def maxpool(pool_size, strides):
  """Small blue block"""
  return mu.MaxPool1D(pool_size, strides)

def dropout():
  """Small grey block"""
  return mu.Dropout(0.5)

def feature_extracting_net(name, n=32):
  return mu.ForkMergeDAG(vertices=[
    [conv1d(50, 2*n), maxpool(8, 8), dropout(), conv1d(4, 4*n),
     conv1d(4, 4*n), conv1d(4, 4*n), conv1d(4, 4*n), maxpool(8, 8)],
    [conv1d(400, 2*n), maxpool(4, 8), dropout(), conv1d(8, 4*n),
     conv1d(8, 4*n), conv1d(8, 4*n), conv1d(8, 4*n), maxpool(2, 8)],
    [mu.Merge.Sum(), dropout()],
    [conv1d(6, n), conv1d(6, 2*n), conv1d(6, 4*n), dropout()],
    [mu.Merge.Sum()]], edges='1;10;011;0001;00011', name=name)

# endregion: Blocks


def get_data_fusion_model():
  model = init_model()
  model.add(feature_extracting_net('DAG'))
  return output_and_build(model)


def get_feature_fusion_model():
  from tframe.nets.octopus import Octopus
  from slp_core import th

  model = init_model()
  oc: Octopus = model.add(Octopus())

  assert len(th.fusion_channels) == 2
  n = 23

  # Input 1
  c = len(th.fusion_channels[0])
  li = oc.init_a_limb('input-1', [3000, c])
  li.add(feature_extracting_net('DAG-1', n=n))

  # Input 2
  c = len(th.fusion_channels[1])
  li = oc.init_a_limb('input-2', [3000, c])
  li.add(feature_extracting_net('DAG-2', n=n))
  oc.set_gates([1, 1])

  return output_and_build(model)
