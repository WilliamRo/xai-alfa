from tframe.data.augment.img_aug import image_augmentation_processor
from tframe import DataSet
from typing import Tuple
from tframe.utils import misc

import os



class LLLAgent(object):

  @classmethod
  def load(cls) -> Tuple[DataSet]:
    from lll_core import th

    if th.task in (th.Tasks.FMNIST, th.Tasks.MNIST):
      return cls.load_XMNIST()
    elif th.task == th.Tasks.SLEEPEDF:
      return cls.load_SLEEPEDFX()
    else:
      raise KeyError(f'!! Unknown task {th.task}')

  # region: Dataset-specific Methods

  # region: [F]MNIST

  @classmethod
  def load_XMNIST(cls):
    """Load (F)MNIST dataset. Currently, only splitting for mod-1 is supported.

    Setting format
    --------------
      th.data_config = 'w_1,w_2,...,w_n'
      e.g.,
      th.data_config = '2,1,1,1'

    User obligation: sum_i(w_i) should divide 10000

    Return: ((train_1, test_1), (train_2, test_2), ..., (train_N, test_N))
    """
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
    dataset.targets = misc.convert_to_one_hot(
      dataset.targets, dataset[dataset.NUM_CLASSES])

    # Split dataset according to setting
    ws = [int(w) for w in th.data_config.split(',')]
    assert 10000 % sum(ws) == 0

    train_set, test_set = dataset.split(6000, 1000, over_classes=True)

    rs = [w / sum(ws) for w in ws]
    train_splits = [int(r * 6000) for r in rs]
    test_splits = [int(r * 1000) for r in rs]

    train_sets = train_set.split(*train_splits, over_classes=True)
    test_sets = test_set.split(*test_splits, over_classes=True)

    return list(zip(train_sets, test_sets))

  # endregion: [F]MNIST

  # region: Sleep-EDFx

  @classmethod
  def load_SLEEPEDFX(cls):
    from lll_core import th

    from slp.slp_agent import SLPAgent
    from slp.slp_datasets.sleepedfx import SleepEDFx

    # Find 51-SLEEP/data/sleepedfx
    data_dir = os.path.dirname(th.data_dir)   # 80-LLL
    data_dir = os.path.dirname(data_dir)      # xai-alfa
    data_dir = os.path.join(data_dir, '51-SLEEP', 'data', 'sleepedfx')

    data_set: SleepEDFx = SLPAgent.load_as_tframe_data(data_dir, 'sleepedf')
    return data_set.partition_lll()

  # endregion: Sleep-EDFx

  # endregion: Dataset-specific Methods



if __name__ == '__main__':
  from lll_core import th

  th.task = th.Tasks.FMNIST
  th.data_config = '2,1,1,1'

  datasets = LLLAgent.load()
  print()