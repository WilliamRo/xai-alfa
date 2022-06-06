import numpy as np
import os
import pickle

from dsc.dsc_set import DSCSet



class RMLSet(DSCSet):

  SNR_list = np.linspace(-20, 18, 20).astype(int)
  CLASS_NAMES = [

  ]

  @classmethod
  def load_as_tframe_data(cls, data_dir):
    data_dict = cls.load_raw_data(data_dir)


    return DSCSet(None)


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


if __name__ == '__main__':
  print(RMLSet.SNR_list)


