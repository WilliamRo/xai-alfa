import numpy as np
import os
import pickle

from dsc.dsc_set import DSCSet



class RMLSet(DSCSet):

  SNR_LIST = np.linspace(-20, 18, 20).astype(int)
  CLASS_NAMES = ['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM', 'BFSK', 'CPFSK',
                 'PAM4', 'WBFM', 'AM-SSB', 'AM-DSB']
  NUM_CLASSES = len(CLASS_NAMES)

  # region: Abstract Methods

  def configure(self, config_string: str):
    pass

  @classmethod
  def load_as_tframe_data(cls, data_dir):
    # Load raw data as data_dict, with a format of each items:
    #   ('<class-name>', SNR): np.ndarray of shape (1000, 2, 128)
    raw_data = cls.load_raw_data(data_dir)

    return RMLSet(name='RML2016.10a', n_to_one=True, raw_data=raw_data)

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

  def _check_data(self):
    key = 'raw_data'
    assert key in self.properties
    assert len(self[key]) == 220

  # endregion: Abstract Methods




if __name__ == '__main__':
  print(RMLSet.SNR_LIST)


