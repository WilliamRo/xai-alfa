from fmnist.fm_agent import FMNIST
from tframe.data.images.mnist import MNIST
from tframe.data.augment.img_aug import image_augmentation_processor
from tframe import DataSet
from typing import Tuple

import os



class LLLAgent(object):

  @classmethod
  def load(cls) -> Tuple[DataSet]:
    from lll_core import th

    data_dir = th.data_dir
    if 'fmnist' in th.data_config:
      Agent = FMNIST
      data_dir = os.path.join(data_dir, 'fmnist')
    else:
      assert 'mnist' in th.data_config
      Agent = MNIST
      data_dir = os.path.join(data_dir, 'mnist')

    # Load data
    train_set, val_set, test_set = Agent.load(data_dir)

    # Set batch_preprocessor for augmentation if required
    if th.augmentation:
      train_set.batch_preprocessor = image_augmentation_processor

    return train_set, val_set, test_set