"""This script provides minimal example for preparing tframe.DataSet of
aligned sequence-to-sequence problems such as signal denoise problems"""
from tframe import DataSet

import numpy as np



# -----------------------------------------------------------------------------
# Create a simple dataset
# -----------------------------------------------------------------------------
def load_as_numpy_arrays(L):
  x = np.random.rand(L)
  y = np.sin(x)
  return x, y

# Generate a sequence set
features, targets = load_as_numpy_arrays(1000)
data_set = DataSet(features, targets, name='Sin Set')

# -----------------------------------------------------------------------------
# See what is fed into an RNN at each iteration
# -----------------------------------------------------------------------------
batch_size = 3
num_steps = 50
for i, batch in enumerate(data_set.gen_rnn_batches(
    batch_size=batch_size, num_steps=num_steps)):
  x, y = batch.features, batch.targets
  print(f'[{i + 1:02}] features: {x.shape}; targets: {y.shape}')
  assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)


# -----------------------------------------------------------------------------
# Create a simple model and do prediction
# -----------------------------------------------------------------------------
from tframe import hub as th
from tframe import Predictor
from tframe import mu
from tframe.models import Recurrent

model = Predictor(mark='mascot', net_type=Recurrent)
model.add(mu.Input(sample_shape=[1]))
model.add(mu.Dense(1))

model.build()

# Predict use model
output = model.predict(data_set, num_steps=30)
print()


