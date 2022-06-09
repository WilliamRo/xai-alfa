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


  def report(self):
    raise NotImplementedError


  def format_data(self):
    from dsc_core import th

    # Set targets
    if th.use_rnn: self.summ_dict[self.TARGETS] = np.expand_dims(
      self.data_dict.pop(self.TARGETS), 1)

  def partition(self):
    from dsc_core import th

    val_size, test_size = th.val_size, th.test_size

    # Calculate val/test size if proportion is provided
    if th.val_proportion is not None:
      assert th.test_proportion is not None
      assert th.val_proportion + th.test_proportion < 1

      group_size = len(self.groups[0])
      val_size = int(group_size * th.val_proportion)
      test_size = int(group_size * th.test_proportion)

    return self.split(-1, val_size, test_size, over_classes=True,
                      random=True, names=('TrainSet', 'ValSet', 'TestSet'))

