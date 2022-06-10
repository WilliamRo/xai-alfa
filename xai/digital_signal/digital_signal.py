from roma import Nomear

import numpy as np



class DigitalSignal(Nomear):
  """This class provides basic abstraction for multi-channel digital signals
  recorded simultaneously. """

  def __init__(self):
    pass

  # region: Properties
  # endregion: Properties

  # region: Public Methods
  # endregion: Public Methods

  # region: Private Methods
  # endregion: Private Methods



if __name__ == '__main__':
  from dsc_core import th
  from dsc.dsc_agent import DSCAgent
  from dsc.dsc_datasets.rml import RMLSet

  th.data_config = 'rml:10;iq'
  ds, _, _ = DSCAgent.load(th.data_dir)
  assert isinstance(ds, RMLSet)

  ds.report()


  # d = DigitalSignal()
  # print()
