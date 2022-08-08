from tframe import DataSet
from lll.lll_agent import LLLAgent

import os



def load_data():
  from lll_core import th

  # Load data
  datasets: list = list(LLLAgent.load())

  for i, data_tuple in enumerate(datasets):
    train_set, test_set = data_tuple
    assert isinstance(train_set, DataSet)
    train_set, val_set = train_set.split(
      -1, int(th.val_pct * train_set.size / 10), over_classes=True)

    train_set.name = f'Train-{i+1}'
    val_set.name = f'Val-{i+1}'
    test_set.name = f'Test-{i+1}'

    datasets[i] = (train_set, val_set, test_set)

  return datasets



if __name__ == '__main__':
  from lll_core import th

  th.task = th.Tasks.FMNIST
  th.data_config = '2,1,1,1'

  datasets = load_data()
  print()




