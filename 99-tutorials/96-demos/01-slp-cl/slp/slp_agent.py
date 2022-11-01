import numpy as np
import os

from tframe import console
from tframe.data.base_classes import DataAgent
from typing import List
from slp.slp_set import SleepSet
from pictor.objects.signals.signal_group import DigitalSignal, SignalGroup

from roma.spqr.finder import walk
from roma import console
from roma import io



class SLPAgent(DataAgent):

  @classmethod
  def load(cls, data_dir, **kwargs):
    # 1. Read raw data
    raw_set: SleepSet = cls.load_as_tframe_data(data_dir)

    # 2. Generate patient groups: [sleep_set_1, sleep_set_2, ...]
    data_sets: List[SleepSet] = raw_set.partition()

    # checkpoint of 2
    # data_sets[1].show()
    # assert False

    # 3. Data config
    # (1) Select normalized channels
    # (2) Convert to AASM format, augment if necessary
    #     output [N, 3000, C]
    # (3) Convert target to onehot format
    console.show_status(f'Configure data ...')
    for ds in data_sets: ds.configure()

    # 4. Split each data_set to (train, val, test)
    return [ds.split(
      7, 1, 2,
      names=[f'Group{i+1}-{s}' for s in ('Train', 'Val', 'Test')],
      over_classes=True) for i, ds in enumerate(data_sets)]

  # region: Raw Data Loading

  @classmethod
  def load_as_tframe_data(cls, data_dir, **kwargs) -> SleepSet:
    tfd_path = os.path.join(data_dir, f'sleepedf-raw.tfd')

    # Load .tfd file directly if it exists
    if os.path.exists(tfd_path): return SleepSet.load(tfd_path)

    # Otherwise, wrap raw data into tframe data and save
    console.show_status(f'Loading raw data from `{data_dir}` ...')

    signal_groups = cls.load_raw_data(data_dir)
    data_set = SleepSet(name=f'Sleep-EDF-Expanded', signal_groups=signal_groups)
    data_set.remove_wake_signal()

    # Save and return
    data_set.save(tfd_path)
    console.show_status(f'Dataset saved to `{tfd_path}`')

    return data_set

  @classmethod
  def load_raw_data(cls, data_dir):
    # Sanity check
    assert os.path.exists(data_dir)

    # Create an empty list
    sleep_groups: List[SignalGroup] = []

    # Get all .edf files
    hypnogram_file_list: List[str] = walk(data_dir, 'file', '*Hypnogram*')
    N = len(hypnogram_file_list)
    print("patient_num:", N)

    # Read records in order
    for i, hypnogram_file in enumerate(hypnogram_file_list):
      # Get id
      id: str = os.path.split(hypnogram_file)[-1].split('-')[0]

      # Get detail
      detail_dict = {}

      # If the corresponding .rec file exists, read it directly
      xai_rec_path = os.path.join(data_dir, id + '.xrec')
      if os.path.exists(xai_rec_path):
        console.show_status(
          f'Loading `{id}` from {data_dir} ...', prompt=f'[{i + 1}' f'/{N}]')
        console.print_progress(i, N)
        sg = io.load_file(xai_rec_path)
        sleep_groups.append(sg)
        continue

      console.show_status(f'Reading record `{id}` ...', prompt=f'[{i+1}/{N}]')
      console.print_progress(i, N)

      # (1) Read PSG file
      fn = os.path.join(data_dir, id[:7] + '0' + '-PSG.edf')
      assert os.path.exists(fn)
      digital_signals: List[DigitalSignal] = cls.read_edf_file(fn)

      # (2) Read stage labels
      labels = {'Sleep stage W':0, 'Sleep stage R':1, 'Sleep stage 1':2,
                'Sleep stage 2':3, 'Sleep stage 3':4, 'Sleep stage 4':5,
                'Movement time':6, 'Sleep stage ?':7}
      stages_ann = cls.read_edf_anno_file_using_mne(hypnogram_file)
      stages = [labels[stage] for stage in stages_ann]
      stages = np.array(stages)

      # .. sanity check (for sleepEDF, '-1' is necessary. error will occur
      # otherwise)
      L = (digital_signals[0].length) // 3000
      assert len(stages) == L

      # Wrap data into signal group
      sg = SignalGroup(digital_signals, label=f'{id}', **detail_dict)
      sg.set_annotation(SleepSet.STAGE_KEY, 30, stages, SleepSet.STAGE_LABELS)
      sleep_groups.append(sg)

      # Save sg if necessary
      save_xai_rec = True
      if save_xai_rec:
        console.show_status(f'Saving `{id}` to `{data_dir}` ...')
        console.print_progress(i, N)
        io.save_file(sg, xai_rec_path)

    console.show_status(f'Successfully read {N} records')
    return sleep_groups


  @classmethod
  def read_edf_file(cls, fn: str, channel_list: List[str] = None) -> List[DigitalSignal]:
    """Read .edf file using pyedflib package.

    :param fn: file name
    :param channel_list: list of channels. None by default.
    :return: a list of DigitalSignals
    """
    import pyedflib

    # Sanity check
    assert os.path.exists(fn)

    signal_dict = {}
    with pyedflib.EdfReader(fn) as file:
      # Check channels
      all_channels = file.getSignalLabels()
      if channel_list is None: channel_list = all_channels
      # Read channels
      for channel_name in channel_list:
        # Get channel id
        chn = all_channels.index(channel_name)
        frequency = file.getSampleFrequency(chn) / 30

        # Initialize an item in signal_dict if necessary
        if frequency not in signal_dict: signal_dict[frequency] = []
        # Read signal
        signal_dict[frequency].append((channel_name, file.readSignal(chn)))

    # Wrap data into DigitalSignals
    digital_signals = []
    for frequency, signal_list in signal_dict.items():
      ticks = np.arange(len(signal_list[0][1])) / frequency
      digital_signals.append(DigitalSignal(
        np.stack([x for _, x in signal_list], axis=-1), ticks=ticks,
        channel_names=[name for name, _ in signal_list],
        label=f'Freq=' f'{frequency}'))

    return digital_signals


  @classmethod
  def read_edf_anno_file_using_mne(cls, fn: str, allow_rename=True)-> list:
    from mne import read_annotations

    # Check extension
    if fn[-3:] != 'edf':
      if not allow_rename:
        # Rename .rec file if necessary, since mne package works only for
        # files with .rec extension
        raise TypeError(f'!! extension of `{fn}` is not .edf')
      os.rename(fn, fn + '.edf')
      fn = fn + '.eef'

    assert os.path.exists(fn)

    stage_anno = []
    raw_anno = read_annotations(fn)
    anno = raw_anno.to_data_frame().values
    anno_dura = anno[:, 1]
    anno_desc = anno[:, 2]
    for dura_num in range(len(anno_dura) - 1):
      for stage_num in range(int(anno_dura[dura_num]) // 30):
        stage_anno.append(anno_desc[dura_num])
    return stage_anno

  # endregion: Raw Data Loading



if __name__ == '__main__':
  from slp_core import th
  th.data_config = 'sleepedf:0,1,2'
  dataset = SLPAgent.load(th.data_dir, suffix='-alpha')
  print()
