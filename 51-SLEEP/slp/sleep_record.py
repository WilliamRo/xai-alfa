from collections import OrderedDict
from roma import console
from roma.spqr.finder import walk
from typing import List

import numpy as np
import os
import pickle



class SleepRecord(object):
  """
  Data modalities:
  (1) EEG: EEG signals, usually has multiple channel, stored in .edf files;
           read as numpy arrays of shape [num_channels, seq_len]

  """

  class Keys:
    EEG = 'EEG'
    STAGES = 'STAGES'

  def __init__(self, record_id: str):
    self.record_id = record_id
    self.data_dict = OrderedDict()

  # region: Public Mehtods

  def report_detail(self):
    console.show_info(f'{self.record_id}:')
    for k, v in self.data_dict.items():
      console.supplement(f'{k}.shape = {v.shape}')

  # endregion: Public Mehtods

  # region: Data IO

  @classmethod
  def read_sleep_records(cls, data_dir: str):
    # Sanity check
    assert os.path.exists(data_dir)

    # If .recs file exists, load it directly
    folder_name = os.path.split(data_dir)[-1]
    recs_fn = os.path.join(data_dir, folder_name + '-xai.recs')
    if os.path.exists(recs_fn):
      # with open support closing file in any situation
      with open(recs_fn, 'rb') as file:
        console.show_status('Loading `{}` ...'.format(recs_fn))
        return pickle.load(file)

    # Otherwise, read sleep records according to folder name
    if folder_name in ['ucddb']:
      sleep_records = cls._read_ucddb(data_dir)
    else:
      raise KeyError(f'!! Unknown dataset `{folder_name}`')

    # Save file and return
    with open(recs_fn, 'wb') as file:
      pickle.dump(sleep_records, file, pickle.HIGHEST_PROTOCOL)
    return sleep_records

  # region: Builtin IO Methods

  @classmethod
  def _read_edf_file(cls, fn: str, channel_list: List[str]) -> np.ndarray:
    from mne.io import concatenate_raws, read_raw_edf
    from mne.io.edf.edf import RawEDF

    with read_raw_edf(fn, preload=True) as raw_edf:
      assert isinstance(raw_edf, RawEDF)
      # edf_data[:, 0] is time, edf_data[:, k] is channel k (k >= 1)
      edf_data = raw_edf.pick_channels(channel_list).to_data_frame().values

    return edf_data

  @classmethod
  def _read_ucddb(cls, data_dir: str):
    # Define static variables
    class Channels:
      EEG = {'SIGNAl_EEG1': "C3A2",
             'SIGNAl_EEG2': "C4A1" }
      ECG = {'SIGNAL_ECG1': "chan 1",
             'SIGNAL_ECG2': "chan 2",
             'SIGNAL_ECG3': "chan 3",
             'SIGNAL_ECG4': "ECG"}

    console.show_status('Reading `ucddb` dataset ...')

    # Create an empty list
    sleep_records: List[cls] = []

    # Get all .edf files
    rec_file_list: List[str] = walk(data_dir, 'file', '*.rec*')

    # Read records one by one
    for rec_fn in rec_file_list:
      # Get id
      id = os.path.split(rec_fn)[-1].split('.r')[0]
      sr = SleepRecord(id)

      console.show_status(f'Reading record `{id}` ...')
      console.split()

      # (1) Read EEG data
      # Rename .rec file if necessary
      if rec_fn[-3:] != 'edf':
        os.rename(rec_fn, rec_fn + '.edf')
        rec_fn = rec_fn + '.edf'

      sr.data_dict['EEG'] = cls._read_edf_file(
        rec_fn, list(Channels.EEG.values()))
      console.split()

      # (2) Read ECG data
      fn = os.path.join(data_dir, id + '_lifecard.edf')
      sr.data_dict['ECG'] = cls._read_edf_file(fn, list(Channels.ECG.values()))
      console.split()

      # (3) Read stage labels
      fn = os.path.join(data_dir, id + '_stage.txt')
      assert os.path.exists(fn)
      with open(fn, 'r') as stage:
        stage_ann = [int(line.strip()) for line in stage.readlines()]
      sr.data_dict['stage'] = np.array(stage_ann)

      # (4) Read misc data
      fn = os.path.join(data_dir, id + '_respevt.txt')

      # Append this record to list
      sleep_records.append(sr)

    # Return records
    return sleep_records

  # endregion: Builtin IO Methods

  # endregion: Data IO



if __name__ == '__main__':
  # Get data path
  abs_path = os.path.abspath(__file__)
  data_dir = os.path.join(os.path.dirname(os.path.dirname(abs_path)),
                          'data', 'ucddb')

  #
  sleep_records = SleepRecord.read_sleep_records(data_dir)
  for slp_rec in sleep_records: slp_rec.report_detail()



