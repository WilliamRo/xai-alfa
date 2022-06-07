from tframe.data.sequences.seq_set import SequenceSet

import numpy as np



class DSCSet(SequenceSet):

  class DataFormat:
    IQ = 'SDCSet.DataFormat.IQ'
    amplitude_and_phase = 'SDCSet.DataFormat.A&P'


  @classmethod
  def load_as_tframe_data(cls, data_dir):
    raise NotImplementedError


  @classmethod
  def load_raw_data(cls, data_dir):
    raise NotImplementedError


  def configure(self, config_string: str):
    raise NotImplementedError


  def format_data(self):
    from dsc_core import th

    # Set targets
    if th.use_rnn: self.summ_dict[self.TARGETS] = np.expand_dims(
      self.data_dict.pop(self.TARGETS), 1)

