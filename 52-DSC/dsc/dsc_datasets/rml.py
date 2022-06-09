import numpy as np
import os
import pickle

from dsc.dsc_set import DSCSet
from tframe.utils import misc
from tframe import console
from tframe import pedia



class RMLSet(DSCSet):

  class Keys:
    raw_data = 'raw_data'
    SNRs = 'SNRs'

  SNR_LIST = np.linspace(-20, 18, 20).astype(int)
  CLASS_NAMES = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64', 'GFSK', 'CPFSK',
                 'PAM4', 'WBFM', 'AM-SSB', 'AM-DSB']

  # region: Properties

  @DSCSet.property()
  def SNRs(self):
    return list(sorted(set(self[self.Keys.SNRs])))

  # endregion: Properties

  # region: Abstract Methods

  def configure(self, config_string: str):
    """config_string should be <snr_config>;<channels>
       <snr_config> should be
         (1)'*': load all data,
         (2) db1[,db2,...]: load signals which have a SNR of db1 (or db2 ...),
             e.g., '10,12'
         (3) db1-db2: load signals with SNR between db1 and db2
       <channels> should be an non-empty string consisting of non-repeat
         chars in ('i', 'q', 'a', 'p'), 'i'/'q' represent complex channels,
         'a' represents amplitude, and 'p' represents phase.
         e.g., 'iq', 'ap'
    """
    snr_config, channels_config = config_string.split(';')

    # Find SNRs to load by parsing config_string
    if snr_config in ('*', 'all'): snr_to_load = self.SNR_LIST
    elif '-' in snr_config:
      db_range = snr_config.split('-')
      min_db = min(self.SNR_LIST) if db_range[0] == '' else int(db_range[0])
      max_db = max(self.SNR_LIST) if db_range[1] == '' else int(db_range[1])
      snr_to_load = [snr for snr in self.SNR_LIST if min_db <= snr <= max_db]
    else: snr_to_load = [int(db) for db in snr_config.split(',')]

    # Check channels
    assert 0 < len(channels_config) < 5
    for c in channels_config: assert c in ('i', 'q', 'a', 'p')

    # Set data_dict and SNRs
    data: dict = self[self.Keys.raw_data]
    IQ_data, targets, SNRs = [], [], []
    for snr in snr_to_load:
      for i, modulation in enumerate(self.CLASS_NAMES):
        array = data[(modulation, snr)]
        IQ_data.append(array)
        targets.extend([i] * len(array))
        SNRs.extend([snr] * len(array))

    # Generate features
    channels_config = channels_config.lower()
    IQ_data = np.concatenate(IQ_data, axis=0)
    I, Q = IQ_data[:, 0], IQ_data[:, 1]
    channels = []
    if 'i' in channels_config: channels.append(I)
    if 'q' in channels_config: channels.append(Q)
    if 'a' in channels_config: channels.append(np.sqrt(I ** 2 + Q ** 2))
    if 'p' in channels_config: channels.append(np.arctan(Q / I))

    # Generate targets
    one_hot_targets = misc.convert_to_one_hot(targets, self.num_classes)

    # Set data
    self.features = np.stack(channels, axis=-1)
    self.targets = one_hot_targets
    self.properties[self.Keys.SNRs] = SNRs

    # Delete raw data in property
    self.properties.pop(self.Keys.raw_data)

  @classmethod
  def load_as_tframe_data(cls, data_dir):
    # Load raw data as data_dict, with a format of each items:
    #   ('<class-name>', SNR): np.ndarray of shape (1000, 2, 128)
    raw_data = cls.load_raw_data(data_dir)
    data_set = RMLSet(name='RML2016.10a', n_to_one=True, raw_data=raw_data)
    data_set.properties[cls.NUM_CLASSES] = len(cls.CLASS_NAMES)
    data_set.properties[pedia.classes] = cls.CLASS_NAMES
    return data_set

  @classmethod
  def load_raw_data(cls, data_dir):
    """Raw data should be placed at `52-DSC/data/rml2016` folder"""
    data_dir = os.path.join(data_dir, 'rml2016')
    file_name = 'RML2016.10a_dict.pkl'
    file_path = os.path.join(data_dir, file_name)
    # Make sure data file exists
    if not os.path.exists(file_path):
      raise FileExistsError(f'!! Cannot find `{file_name}` under `{data_dir}`')

    # Load raw data dict and return
    with open(file_path, 'rb') as f: return pickle.load(f, encoding='latin-1')

  def report(self):
    from dsc_core import th
    from tframe.utils.display.table import Table

    console.show_info(f'{self.name} Detail')
    console.supplement(f'Shape of features: {self.features.shape}')
    console.supplement(f'Shape of targets: {self[self.TARGETS].shape}')

    if not th.report_detail:
      console.supplement(f'Length of groups: {[len(g) for g in self.groups]}')
      return

    # Initialize a Table
    width = max([len(s) for s in self.CLASS_NAMES])
    t = Table(*([3] + [width] * len(self.CLASS_NAMES)))
    t.print_header('', *self.CLASS_NAMES)

    m = np.zeros(shape=(len(self.SNRs), len(self.CLASS_NAMES)), dtype=int)
    for j, indices in enumerate(self.groups):
      for i, snr in enumerate(self.SNRs):
        m[i, j] = sum(np.equal(snr, [self[self.Keys.SNRs][ind]
                                     for ind in indices]))

    # Print
    for r, snr in zip(m, self.SNRs): t.print_row(snr, *r)
    t.hline()
    t.print_row('Sum', *[len(g) for g in self.groups])
    t.hline()

  # endregion: Abstract Methods

  # region: Overwriting

  def partition_(self):
    from dsc_core import th

    val_p, test_p = th.val_proportion, th.test_proportion
    assert 0 < val_p + test_p < 1

    # TODO:


  def _check_data(self):
    """This method will be called during splitting dataset"""
    if self.Keys.raw_data in self.properties:
      assert len(self[self.Keys.raw_data]) == 220

  # endregion: Overwriting

  # region: Data Visualization

  def show(self):
    pass

  # endregion: Data Visualization



if __name__ == '__main__':
  pass


