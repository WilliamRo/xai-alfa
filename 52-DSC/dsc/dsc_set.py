from tframe.data.sequences.seq_set import SequenceSet



class DSCSet(SequenceSet):

  class DataFormat:
    IQ = 'SDCSet.DataFormat.IQ'


  @classmethod
  def load_as_tframe_data(cls, data_dir):
    raise NotImplementedError


  @classmethod
  def load_raw_data(cls, data_dir):
    raise NotImplementedError



