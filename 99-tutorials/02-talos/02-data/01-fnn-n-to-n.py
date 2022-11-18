from tframe.data.dataset import DataSet

import numpy as np



# -----------------------------------------------------------------------------
# Create a simple dataset
# -----------------------------------------------------------------------------
def load_as_numpy_arrays(n):
  features, targets = [], []
  for _ in range(n):
    L = 32
    x = np.random.rand(L)
    features.append(x)
    targets.append(np.sin(x))

  features, targets = np.stack(features), np.stack(targets)
  features = np.reshape(features, list(features.shape) + [1])
  targets = np.reshape(targets, list(targets.shape) + [1])
  return features, targets


# Generate a sequence set
features, targets = load_as_numpy_arrays(20)
data_set = DataSet(features, targets, name='Sin Set')

# -----------------------------------------------------------------------------
# Create a simple model and do prediction
# -----------------------------------------------------------------------------
from tframe import hub as th
from tframe import Predictor
from tframe import mu

model = Predictor(mark='mascot')
model.add(mu.Input(sample_shape=[32, 1]))
model.add(mu.HyperConv1D(4, 3, 4))
model.add(mu.HyperDeconv1D(2, 4, 2))
model.add(mu.HyperDeconv1D(1, 5, 2))

model.build()

# Predict use model
output = model.predict(data_set)
print()