from tframe import DataSet
from tframe.data.images.mnist import MNIST
from tframe import hub as th

import numpy as np



def load_data(path):
  train_set, val_set, test_set = MNIST.load(
    path, validate_size=5000, test_size=10000, flatten=False, one_hot=True)
  assert isinstance(train_set, DataSet)
  assert isinstance(val_set, DataSet)
  assert isinstance(test_set, DataSet)

  if 'macro' in th.developer_code:
    train_set.gen_batches = lambda *args, **kwargs: gen_batches(
      train_set, *args, **kwargs)

  return train_set, val_set, test_set



def gen_batches(self, batch_size, shuffle=False, is_training=False):

  round_len = self.get_round_length(batch_size, training=is_training)
  if batch_size == -1: batch_size = self.size

  # Generate batches
  # !! Without `if is_training:`, error will occur if th.validate_train_set
  #    is on
  if is_training: self._init_indices(shuffle)
  for i in range(round_len):
    indices = self._select(i, batch_size, training=is_training)

    if is_training:
      pass
      # ---------------------------------- (1)
      # C = np.random.randint(0, 10)
      # N = int(len(indices) * P)
      #
      # indices[:N] = np.random.choice(self.groups[C], N)
      # ---------------------------------- (1) end

    # Get subset
    data_batch = self[indices]

    if is_training and len(indices) == batch_size:
      from tframe import hub as th
      # ---------------------------------- (2)
      P = th.alpha
      BS = batch_size
      N = int(BS * P)
      if N > 0:
        data_batch.targets[BS-N:] = data_batch.targets[:N]
      # ---------------------------------- (2) end

    # Preprocess if necessary
    if self.batch_preprocessor is not None:
      data_batch = self.batch_preprocessor(data_batch, is_training)
    # Make sure data_batch is a regular array
    if not data_batch.is_regular_array: data_batch = data_batch.stack
    # Yield data batch
    yield data_batch

  # Clear dynamic_round_len if necessary
  if is_training: self._clear_dynamic_round_len()




if __name__ == '__main__':
  from tframe.data.images.image_viewer import ImageViewer
  from mn_core import th
  train_set, val_set, test_set = load_data(th.data_dir)
  viewer = ImageViewer(test_set)
  viewer.show()
