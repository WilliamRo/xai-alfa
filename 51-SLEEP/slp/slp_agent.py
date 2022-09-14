import numpy as np

from tframe import console
from tframe.data.base_classes import DataAgent
from slp.slp_set import SleepSet

import os



class SLPAgent(DataAgent):
  """To use this class, th.data_config must be specified.
  The syntax is `<...>:<...>`, e.g.,
  """

  @classmethod
  def load(cls, data_dir, **kwargs):
    from dsc_core import th
    data_name, config_string = th.data_config.split(':')

    # Load DSCSet
    data_set = cls.load_as_tframe_data(data_dir, data_name, **kwargs)
    data_set.configure(config_string)

    # Format data
    data_set.format_data()

    return data_set

    # Split and return
    # return data_set.partition()


  @classmethod
  def load_as_tframe_data(cls, data_dir, data_name = None, **kwargs) -> SleepSet:
    # file_path = cls._get_tfd_file_path(data_dir, data_name, suffix=suffix, **kwargs)
    # if os.path.exists(file_path): return SleepSet.load(file_path)

    # If dataset does not exist, create a new one, save and return
    if data_name == 'ucddb':
      from slp.slp_datasets.ucddb import UCDDB as DataSet
    elif data_name in ['sleep-cassette-train20','sleepedf-lll']:
      from slp.slp_datasets.sleepedfx import SleepEDFx as DataSet
    else: raise KeyError(f'!! Unknown dataset `{data_name}`')

    console.show_status(f'Loading `{data_name}` from {data_dir}')
    data_set = DataSet.load_as_tframe_data(data_dir, data_name, **kwargs)
    # data_set.save(file_path)
    # console.show_status(f'Dataset saved to `{file_path}`')
    return data_set


  @classmethod
  def _get_tfd_file_path(cls, data_dir, data_name, **kwargs):
    suffix = kwargs['suffix']
    return os.path.join(data_dir, data_name, f'{data_name}{suffix}.tfds')



if __name__ == '__main__':
  from slp_core import th
  th.data_config = 'sleepedf:0,1,2'
  dataset = SLPAgent.load(th.data_dir, suffix='-alpha')
  print()
