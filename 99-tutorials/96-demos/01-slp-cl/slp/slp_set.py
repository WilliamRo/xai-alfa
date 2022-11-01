from typing import List
from tframe import DataSet
from tframe.utils.misc import convert_to_one_hot
from pictor.objects.signals.signal_group import SignalGroup
from roma import console

import numpy as np



class SleepSet(DataSet):

  STAGE_KEY = 'STAGE'
  STAGE_LABELS = ['Wake', 'REM', 'Stage 1', 'Stage 2', 'Stage 3', 'Stage 4',
                  'Movement', 'Indeterminate']
  CHANNELS = {'0': 'EEG Fpz-Cz',
              '1': 'EEG Pz-Oz',
              '2': 'EOG horizontal',
              '3': 'Resp oro-nasal',
              '4': 'EMG submental',
              '5': 'Temp rectal',
              '6': 'Event marker'}

  # region: Properties

  @property
  def signal_groups(self) -> List[SignalGroup]:
    return self.properties['signal_groups']

  @signal_groups.setter
  def signal_groups(self, value):
    self.properties['signal_groups'] = value

  # endregion: Properties

  # region: Data Config

  def partition(self):
    """Returns a list of SleepSet [data_set1, data_set2, ...]"""
    from slp_core import th

    data_sets = []
    cursor = 0
    for i, str_n in enumerate(th.data_config.split(',')):
      n = int(str_n)
      ds = SleepSet(name=f'Group-{i+1}')
      ds.signal_groups = self.signal_groups[cursor:cursor + n]
      data_sets.append(ds)
      cursor += n

    return data_sets

  # endregion: Data Config

  # region: Configure

  def configure(self):
    """config_string examples: '0,2,6'
       (1) Select normalized channels
       (2) Convert to AASM format, augment if necessary
           output [N, 3000, C]
       (3) Split channel if necessary
           (3.1) for data-fusion,
                 self.data_dict['features'].shape = [N, 3000, C]
           (3.2) for feature-fusion, e.g., th.channels = '0,1;2'
                 self.data_dict['input-1'].shape = [N, 3000, 2]
                 self.data_dict['input-2'].shape = [N, 3000, 1]

    Augmentation format:
        0: Wake, 1: REM, 2: N1, 3: N2, 4: N3
        Syntax: i_1*n_1+i_2*n_2...
    e.g., 2*3+4*3 represents 'N1*3+N3*3'
    """
    from slp_core import th

    # Parse aug_config, e.g., {2: 3, 4: 3}
    delta = 400
    aug_config = {}
    for s in th.aug_config.split('+'):
      if s == '': break
      assert s[1] == '*'
      aug_config[int(s[0])] = int(s[2])

    # region: Utility Functions

    def data_preprocess(data):
      import numpy as np
      from scipy import signal
      # Filter data
      b, a = signal.butter(7, 0.7, 'lowpass')
      filtered_data = signal.filtfilt(b, a, data)
      # Normalize data
      arr_mean = np.mean(filtered_data)
      arr_std = np.std(filtered_data)
      precessed_data = (filtered_data - arr_mean) / arr_std
      return precessed_data

    # endregion: Utility Functions

    # Get channel indices
    channel_names = [self.CHANNELS[s] for s in th.channel_list]

    features, targets = [], []
    for sg in self.signal_groups:
      # (1) Get data of shape [L, C]
      sequence = np.stack([data_preprocess(sg[name]) for name in channel_names], axis=-1)
      annotations = sg.annotations[self.STAGE_KEY].annotations

      # (2) Convert to AASM format, augment if required
      for k, label in enumerate(annotations):
        # Check label first
        if label in [6, 7]: continue
        if label == 5: label = 4

        # Put data into features, augment if necessary
        start_i, end_i = k * 3000, (k + 1) * 3000
        features.append(sequence[start_i:end_i])
        targets.append(label)

        # Augment if necessary
        if label not in aug_config: continue
        for i in range(aug_config[label] - 1):
          d = np.random.randint(-delta, delta)
          if k == 0: d = max(0, d)
          elif k == len(annotations) - 1: d = min(0, d)
          features.append(sequence[start_i+d:end_i+d])
          targets.append(label)

    # Stack features, and convert targets to onehot
    # shape: [[3000, C], [3000, C], ...] => [N, 3000, C]
    self.features = np.stack(features, axis=0)
    self.targets = convert_to_one_hot(targets, 5)
    self.properties['NUM_CLASSES'] = 5

    # (3) Split channel if necessary
    if ';' not in th.channels: return
    for i, channels in enumerate(th.fusion_channels):
      self.data_dict[f'input-{i+1}'] = np.stack(
        [self.features[:, :, int(c)] for c in channels], axis=-1)


  def remove_wake_signal(self):
    # For each patient
    for sg in self.signal_groups:
      # Cut annotations
      annotation = sg.annotations[self.STAGE_KEY]
      non_zero_indice = np.argwhere(annotation.annotations != 0)

      start, end = min(non_zero_indice)[0], max(non_zero_indice)[0]

      margin = 60
      start, end = start - margin, end + margin

      annotation.intervals = annotation.intervals[start:end]
      annotation.annotations = annotation.annotations[start:end]

      for ds in sg.digital_signals:
        freq = int(float(ds.label.split('=')[1]))
        _start, _end = start * freq * 30, end * freq * 30
        ds.ticks = ds.ticks[_start:_end]
        ds.sequence = ds.sequence[_start:_end]

  # endregion: Configure

  # region: MISC

  def _check_data(self): pass

  # endregion: MISC

  # region: Report and visualize

  def report(self):
    console.supplement(f'{self.name} Details:')
    for k, v in self.data_dict.items():
      console.supplement(f'{k} shape = {v.shape}', level=2)
    for i, g in enumerate(self.groups):
      console.supplement(f'{self.STAGE_LABELS[i]}: {len(g)} epochs', level=3)

  def show(self):
    from pictor import Pictor
    from pictor.plotters import Monitor

    # Initialize pictor and set objects
    p = Pictor(title='Sleep-EDFx', figure_size=(12, 8))
    p.objects = self.signal_groups

    # Set monitor
    m: Monitor = p.add_plotter(Monitor())
    m.channel_list = [c for c, _, _ in self.signal_groups[0].name_tick_data_list]

    # .. set annotation logic
    anno_key = 'annotation'
    m.set(anno_key, self.STAGE_KEY)
    def on_press_a():
      if m.get(anno_key) is None: m.set(anno_key, self.STAGE_KEY)
      else: m.set(anno_key)
    m.register_a_shortcut('a', on_press_a, 'Toggle annotation')

    p.show()

  # endregion: Report and visualize


