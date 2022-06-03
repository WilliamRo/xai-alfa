"""This script provides minimal example for preparing tframe.SequenceSet of
aligned sequence-to-sequence problems such as signal denoise problems"""
from tframe.data.sequences.seq_set import SequenceSet

import numpy as np



# -----------------------------------------------------------------------------
# Create a simple dataset
# -----------------------------------------------------------------------------
def load_as_numpy_arrays(n):
  features, targets = [], []
  for _ in range(n):
    L = np.random.randint(20, 30)
    x = np.random.rand(L)
    features.append(x)
    targets.append(np.sin(x))
  return features, targets


# Generate a sequence set
features, targets = load_as_numpy_arrays(20)
data_set = SequenceSet(features, targets, name='Sin Set')

# -----------------------------------------------------------------------------
# See what is fed into an RNN at each iteration
# -----------------------------------------------------------------------------
batch_size = 3
for i, batch in enumerate(data_set.gen_rnn_batches(
    batch_size=batch_size, num_steps=-1)):

  x, y = batch.features, batch.targets
  print(f'[{i:02}] features: {x.shape}; targets: {y.shape}')
  assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)

  # Display first batch
  if i == 0:
    for j, seq in enumerate(x): print(f'batch[{j}]: {seq.flatten()}')

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
output = model.predict(data_set)
print()