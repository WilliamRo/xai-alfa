from tframe.data.augment.img_aug import image_augmentation_processor
from tframe import DataSet
from typing import Tuple

import os



class LLLAgent(object):

  @classmethod
  def load(cls) -> Tuple[DataSet]:
    from lll_core import th

    if th.task in (th.Tasks.FMNIST, th.Tasks.MNIST):
      return cls.load_XMNIST()
    else:
      raise KeyError(f'!! Unknown task {th.task}')

  # region: Dataset-specific Methods

  @classmethod
  def load_XMNIST(cls):
    from fmnist.fm_agent import FMNIST
    from tframe.data.images.mnist import MNIST

    from lll_core import th

    data_dir = th.data_dir
    if th.task is th.Tasks.FMNIST:
      # 7000 samples per class
      Agent = FMNIST
      data_dir = os.path.join(data_dir, 'fmnist')
    else:
      assert th.task is th.Tasks.MNIST
      Agent = MNIST
      data_dir = os.path.join(data_dir, 'mnist')

    # Load whole dataset
    dataset = Agent.load_as_tframe_data(data_dir)

    # Split dataset according to setting


    return (dataset, )

  # endregion: Dataset-specific Methods



if __name__ == '__main__':
  from lll_core import th

  th.task = th.Tasks.FMNIST

  datasets = LLLAgent.load()