from roma import console

import numpy as np
import os



def load_features_and_targets(data_key, normalize=True):
  data_dir = os.path.join(rf'../../data/{data_key}')
  features = np.load(os.path.join(data_dir, 'features.npy'))
  targets = np.load(os.path.join(data_dir, 'targets.npy'))

  console.show_info(f'features.shape = {features.shape}')
  console.show_info(f'targets.shape = {targets.shape}')

  if normalize:
    mu = np.mean(features, axis=0, keepdims=True)
    sigma = np.std(features, axis=0, keepdims=True)
    features = (features - mu) / sigma

  return features, targets
