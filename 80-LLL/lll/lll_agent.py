import numpy as np
import pickle

from tframe.data.augment.img_aug import image_augmentation_processor
from tframe import DataSet
from typing import Tuple
from tframe.utils import misc
from tframe import console
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
      (1) th.data_config = 'alpha:w_1,w_2,...,w_n'
      (2) th.data_config = 'beta:a', here 0 < a < 1
      e.g.,
      th.data_config = 'alpha:2,1,1,1'
      th.data_config = 'beta:0.8'

    User obligation: sum_i(w_i) should divide 10000

    Return: ((train_1, test_1), (train_2, test_2), ..., (train_N, test_N))
    """
    from fmnist.fm_agent import FMNIST
    from tframe.data.images.mnist import MNIST

    from lll_core import th

    # Parse method and arg_str
    method, arg_str = th.data_config.split(':')
    assert method in ('alpha', 'beta')

    data_dir = th.data_dir
    if th.task is th.Tasks.FMNIST:
      # 7000 samples per class
      Agent = FMNIST
      data_dir = os.path.join(data_dir, 'fmnist')
    else:
      assert th.task is th.Tasks.MNIST
      # assert method != 'beta'
      Agent = MNIST
      data_dir = os.path.join(data_dir, 'mnist')

    # Load whole dataset
    dataset = Agent.load_as_tframe_data(data_dir)
    dataset.targets = misc.convert_to_one_hot(
      dataset.targets, dataset[dataset.NUM_CLASSES])

    return {'alpha': cls._XMNIST_alpha,
            'beta': cls._XMNIST_beta}[method](dataset, arg_str)

  @classmethod
  def _XMNIST_alpha(cls, dataset, arg_str: str):
    """This method works only for FMNIST"""
    # Split dataset according to setting
    ws = [int(w) for w in arg_str.split(',')]
    assert 10000 % sum(ws) == 0

    train_set, test_set = dataset.split(6000, 1000, over_classes=True)

    rs = [w / sum(ws) for w in ws]
    train_splits = [int(r * 6000) for r in rs]
    test_splits = [int(r * 1000) for r in rs]

    train_sets = train_set.split(*train_splits, over_classes=True)
    test_sets = test_set.split(*test_splits, over_classes=True)

    return list(zip(train_sets, test_sets))

  @classmethod
  def _XMNIST_beta(cls, dataset: DataSet, arg_str: str):
    """Classes will be divided into 5 groups, namely,
       (0, 1), (2, 3), (4, 5), (6, 7), and (8, 9).
    In each group, samples of two corresponding classes account for p%
    (e.g., 80%), and the rest eight classes account for (1-p)% uniformly.
    """
    # Parse arg_str
    p = float(arg_str)

    # train_set.size = 60000, test_set.size = 10000
    train_set, test_set = dataset.split(6000, 1000, over_classes=True)

    # Put whole test set into depot
    from lll_core import th
    test_set.name = 'Test-*'
    th.depot['test_set'] = test_set

    # Deep-copy groups
    train_groups = [g[:] for g in train_set.groups]
    test_groups = [g[:] for g in test_set.groups]

    # Split data
    data_sets = []

    for i in range(4):
      major_classes = 2 * i, 2 * i + 1
      sub_set = []
      # For i = 0, 1, 2, 3
      for ds, groups in zip((train_set, test_set), (train_groups, test_groups)):
        n_major = int(np.round(p * ds.size / 5 / 2))
        n_minor = int(np.round((1 - p) * ds.size / 5 / 8))

        indices = []
        for c in range(10):
          n = n_major if c in major_classes else n_minor
          indices.extend(groups[c][:n])
          groups[c] = groups[c][n:]

        sub_set.append(ds[indices])

      # Append (train_i, test_i) to data_sets
      data_sets.append(sub_set)

    # Pack remains as last split
    data_sets.append([train_set[np.concatenate(train_groups).astype(np.int)],
                      test_set[np.concatenate(test_groups).astype(np.int)]])

    return data_sets

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
    data_dir = os.path.join(data_dir, '51-SLEEP', 'data')

    data_name = 'sleepedf-lll'
    first_k = sum([int(i) for i in th.data_config.split(',')])
    suffix = '-alpha'
    suffix_k = '' if first_k is None else f'({first_k})'
    tfd_preprocess_path = os.path.join(data_dir, data_name, f'{data_name}{suffix_k}{suffix}-pro.tfds')
    if os.path.exists(tfd_preprocess_path):
      with open(tfd_preprocess_path,'rb') as input_:
        console.show_status('Loading `{}` ...'.format(tfd_preprocess_path))
        return pickle.load(input_)

    data_set: SleepEDFx = SLPAgent.load_as_tframe_data(data_dir,
                                                       data_name=data_name,
                                                       first_k=first_k,
                                                       suffix=suffix)
    return data_set.partition_lll(tfd_preprocess_path)

  # endregion: Sleep-EDFx

  # endregion: Dataset-specific Methods



if __name__ == '__main__':
  from lll_core import th

  th.task = th.Tasks.FMNIST
  th.data_config = 'beta:0.8'

  datasets = LLLAgent.load()

  N = 0
  for ds1, ds2 in datasets: N += ds1.size + ds2.size
  print(N)