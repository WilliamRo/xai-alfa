import numpy as np
import os
import pickle

from slp.slp_set import SleepSet
from tframe.utils import misc
from tframe import console
from tframe import pedia



class UCDDB(SleepSet):

  class Keys:
    pass

  # region: Properties

  # endregion: Properties

  # region: Abstract Methods

  @classmethod
  def load_as_tframe_data(cls, data_dir):
    raise NotImplementedError


  @classmethod
  def load_raw_data(cls, data_dir):
    raise NotImplementedError


  def configure(self, config_string: str):
    raise NotImplementedError


  def report(self):
    raise NotImplementedError

  # endregion: Abstract Methods

  # region: Overwriting

  def _check_data_(self):
    """This method will be called during splitting dataset"""
    pass

  # endregion: Overwriting

  # region: Data Visualization

  def show(self):
    pass

# endregion: Data Visualization



if __name__ == '__main__':
  from slp_core import th
  from slp.slp_agent import SLPAgent

  th.data_config = 'ucddb'
  train_set, val_set, test_set = SLPAgent.load(th.data_dir)
  assert isinstance(train_set, UCDDB)

  train_set.report()
  # train_set.show()

