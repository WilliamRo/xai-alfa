import numpy as np

from tframe import console
from tframe.data.base_classes import DataAgent
from tframe.utils import misc
from dsc.dsc_set import DSCSet

import os



class DSCAgent(DataAgent):
  """To use this class, th.data_config must be specified.
  The syntax is `<data-name>:<config-string>`, e.g.,
    'rml:*' means reading all data in rml2016 dataset;
    'rml:10,18' means reading all data of 10dB and 18dB in rml2016 dataset
  """

  @classmethod
  def load(cls, data_dir, **kwargs):
    from dsc_core import th
    data_name, config_string = th.data_config.split(':')

    # Load DSCSet
    data_set = cls.load_as_tframe_data(data_dir, data_name)
    data_set.configure(config_string)

    # Format data
    data_set.format_data()

    # Split and return
    return data_set.partition()


  @classmethod
  def load_as_tframe_data(cls, data_dir, data_name) -> DSCSet:
    file_path = cls._get_tfd_file_path(data_dir, data_name)
    if os.path.exists(file_path): return DSCSet.load(file_path)

    # If dataset does not exist, create a new one, save and return
    if data_name in ('rml', 'rml2016'):
      from dsc.dsc_datasets.rml import RMLSet as DataSet
    else: raise KeyError(f'!! Unknown dataset `{data_name}`')

    console.show_status(f'Loading `{data_name}` from {data_dir}')
    data_set = DataSet.load_as_tframe_data(data_dir)
    data_set.save(file_path)
    console.show_status(f'Dataset saved to `{file_path}`')
    return data_set


  @classmethod
  def _get_tfd_file_path(cls, data_dir, data_name):
    return os.path.join(data_dir, f'{data_name}.tfds')



if __name__ == '__main__':
  from dsc_core import th
  th.data_config = 'rml:10'
  train_set, val_set, test_set = DSCAgent.load(th.data_dir, 100, 100)
  print()