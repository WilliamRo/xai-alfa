from tframe import console
from tframe import DataSet
from tframe.utils import misc

import numpy as np
import os



def load_data():
  from sie_core import th

  # Load data
  data_dir = os.path.join(th.data_dir, th.data_config)
  features = np.load(os.path.join(data_dir, 'features.npy'))
  targets = np.load(os.path.join(data_dir, 'targets.npy'))
  th.input_shape = [features.shape[-1]]

  # Wrap data into tfr.DataSet
  data_set = DataSet(features, name=th.data_config,
                     NUM_CLASSES=len(set(targets)))
  data_set.targets = misc.convert_to_one_hot(
    targets, data_set.num_classes)
  data_set.report()

  # Split data_set according to K-fold validation setting
  train_set, test_set = data_set.split_k_fold(
    th.folds_k, th.folds_i, over_classes=True)
  test_set.name = 'Test Set'

  # Normalize data
  mu = np.mean(train_set.features, axis=0).reshape([1, -1])
  sigma = np.std(train_set.features, axis=0).reshape([1, -1])
  train_set.features = (train_set.features - mu) / sigma
  test_set.features = (test_set.features - mu) / sigma

  # Split and return
  if th.val_size > 0 and th.train_size > 0:
    train_set, val_set = train_set.split(
      th.train_size, th.val_size, names=['Train Set', 'Val Set'],
      over_classes=True)
    for ds in (train_set, val_set, test_set): ds.report()
    return train_set, val_set, test_set

  for ds in (train_set, test_set): ds.report()
  return train_set, test_set



if __name__ == '__main__':
  from sie_core import th

  th.data_config = 'gordon-424'
  th.folds_i = 1
  th.folds_k = 5

  train_set, test_set = load_data()


