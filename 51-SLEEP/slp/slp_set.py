import os.path
from typing import List
from tframe.data.sequences.seq_set import SequenceSet
from pictor.objects.signals.signal_group import DigitalSignal, SignalGroup

import numpy as np



class SleepSet(SequenceSet):

  STAGE_KEY = 'STAGE'

  # region: Properties

  @property
  def signal_groups(self) -> List[SignalGroup]:
    return self.properties['signal_groups']

  # endregion: Properties

  # region: APIs

  @classmethod
  def load_as_tframe_data(cls, data_dir):
    raise NotImplementedError

  @classmethod
  def load_raw_data(cls, data_dir):
    raise NotImplementedError

  def configure(self, config_string: str):
    """
    config_string examples: '0,2,6'
    """
    raise NotImplementedError

  def report(self):
    raise NotImplementedError

  # endregion: APIs

  # region: Data IO

  @classmethod
  def read_edf_file(cls, fn: str, channel_list: List[str] = None,
                    freq_modifier=None) -> List[DigitalSignal]:
    """Read .edf file using pyedflib package.

    :param fn: file name
    :param channel_list: list of channels. None by default.
    :param freq_modifier: This arg is for datasets such as Sleep-EDF, in which
                          frequency provided is incorrect.
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
        frequency = file.getSampleFrequency(chn)

        # Apply freq_modifier if provided
        if callable(freq_modifier): frequency = freq_modifier(frequency)

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
      fn = fn + '.edf'

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

  @classmethod
  def read_edf_file_using_mne(cls, fn: str, channel_list: List[str],
                              allow_rename=True) -> np.ndarray:
    """Read .edf file using `mne` package"""
    from mne.io import read_raw_edf
    from mne.io.edf.edf import RawEDF

    # Check extension
    if fn[-3:] != 'edf':
      if not allow_rename:
        # Rename .rec file if necessary, since mne package works only for
        # files with .rec extension
        raise TypeError(f'!! extension of `{fn}` is not .edf')
      os.rename(fn, fn + '.edf')
      fn = fn + '.edf'

    assert os.path.exists(fn)

    with read_raw_edf(fn, preload=True) as raw_edf:
      assert isinstance(raw_edf, RawEDF)
      channel_list = list(channel_list)
      edf_data = raw_edf.pick_channels(channel_list).to_data_frame().values

    return edf_data

  # endregion: Data IO

  # region: Data Configuration

  def format_data(self):
    from dsc_core import th

    # Set targets
    if th.use_rnn:
      self.summ_dict[self.TARGETS] = np.expand_dims(
      self.data_dict.pop(self.TARGETS), 1)


  def partition(self):
    raise NotImplementedError
    """TODO: """
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

  # endregion: Data Configuration

  # region: Visualization

  def show(self, channels: List[str] = None, **kwargs):
    from pictor import Pictor
    from pictor.plotters import Monitor

    p = Pictor(title='SleepSet', figure_size=(8, 6))
    p.objects = self.signal_groups
    p.add_plotter(Monitor(**kwargs))
    p.show()

  # endregion: Visualization
