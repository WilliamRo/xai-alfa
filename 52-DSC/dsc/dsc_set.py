from tframe.data.sequences.seq_set import SequenceSet



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



