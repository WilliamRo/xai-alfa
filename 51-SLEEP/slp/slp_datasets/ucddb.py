import numpy as np
import os
import pandas as pd

from pictor.objects.signals.signal_group import DigitalSignal, SignalGroup

from roma.spqr.finder import walk
from roma import console
from roma import io
from slp.slp_set import SleepSet
from typing import List



class UCDDB(SleepSet):
  """This database contains 25 full overnight polysomnograms with simultaneous
  three-channel Holter ECG, from adult subjects with suspected sleep-disordered
  breathing. A revised version of this database was posted on 1 September 2011.
  """

  TICKS_PER_EPOCH = 128 * 30
  STAGE_KEY = 'STAGE'
  STAGE_LABELS = ['Wake', 'REM', 'Stage 1', 'Stage 2', 'Stage 3', 'Stage 4',
                  'Artifact', 'Indeterminate']

  class DetailKeys:
    number = 'Study Number'
    height = 'Height (cm)'
    weight = 'Weight (kg)'
    gender = 'Gender'
    bmi = 'BMI'
    age = 'Age'
    sleepiness_score = 'Epworth Sleepiness Score'
    study_duration = 'Study Duration (hr)'
    sleep_efficiency = 'Sleep Efficiency (%)'
    num_blocks = 'No of data blocks in EDF'

  # region: Properties

  # endregion: Properties

  # region: Abstract Methods (Data IO)

  @classmethod
  def load_as_tframe_data(cls, data_dir, data_name=None, **kwargs) -> SleepSet:
    data_dir = os.path.join(data_dir, 'ucddb')
    tfd_path = os.path.join(data_dir, 'ucddb.tfds')

    # Load .tfd file directly if it exists
    if os.path.exists(tfd_path): return cls.load(tfd_path)

    # Otherwise, wrap raw data into tframe data and save
    console.show_status(f'Loading raw data from `{data_dir}` ...')

    signal_groups = cls.load_raw_data(data_dir, save_xai_rec=True)
    data_set = UCDDB(name='ucddb-1.0.0', signal_groups=signal_groups)
    io.save_file(data_set, tfd_path, verbose=True)
    return data_set


  @classmethod
  def load_raw_data(cls, data_dir, save_xai_rec=False, **kwargs):
    """Load raw data into signal groups. For each subject, four categories of
    data are read:
    (1) EEG
    (2) ECG
    (3) stage
    (4) events
    However, currently, only (1), (3) will be used.
    """
    # Sanity check
    assert os.path.exists(data_dir)

    # Read SubjectDetails.xls
    xls_path = os.path.join(data_dir, 'SubjectDetails.xls')
    df = pd.read_excel(xls_path)

    # Create an empty list
    sleep_groups: List[SignalGroup] = []

    # Get all .edf files
    rec_file_list: List[str] = walk(data_dir, 'file', '*.rec*')
    N = len(rec_file_list)

    # Read records in order
    for i, rec_fn in enumerate(rec_file_list):
      # Get id
      id: str = os.path.split(rec_fn)[-1].split('.r')[0]

      # Get detail
      detail_dict = df.loc[df[cls.DetailKeys.number] == id.upper()].to_dict(
        orient='index').popitem()[1]

      # If the corresponding .rec file exists, read it directly
      xai_rec_path = os.path.join(data_dir, id + '.xrec')
      if os.path.exists(xai_rec_path) and not kwargs.get('overwrite', False):
        console.show_status(
          f'Loading `{id}` from {data_dir} ...', prompt=f'[{i + 1}' f'/{N}]')
        console.print_progress(i, N)
        sg = io.load_file(xai_rec_path)
        sleep_groups.append(sg)
        continue

      console.show_status(f'Reading record `{id}` ...', prompt=f'[{i+1}/{N}]')
      console.print_progress(i, N)

      # (1) Read .rec file
      digital_signals: List[DigitalSignal] = cls.read_edf_file(rec_fn)

      # (2) Read ECG data TODO:
      fn = os.path.join(data_dir, id + '_lifecard.edf')

      # (3) Read stage labels
      fn = os.path.join(data_dir, id + '_stage.txt')
      assert os.path.exists(fn)
      with open(fn, 'r') as stage:
        stage_ann = [int(line.strip()) for line in stage.readlines()]
      stages = np.array(stage_ann)
      stages[stages > 7] = 7
      assert max(stages) <= 7

      # .. sanity check (for ucddb, '-1' is necessary. Error will occur
      # otherwise)
      L = (digital_signals[1].length - 1) // cls.TICKS_PER_EPOCH
      assert len(stages) == L

      # Wrap data into signal group
      sg = SignalGroup(digital_signals, label=f'{id}', **detail_dict)
      sg.set_annotation(cls.STAGE_KEY, 30, stages, cls.STAGE_LABELS)
      sleep_groups.append(sg)

      # Save sg if necessary
      if save_xai_rec:
        console.show_status(f'Saving `{id}` to `{data_dir}` ...')
        console.print_progress(i, N)
        io.save_file(sg, xai_rec_path)

    console.show_status(f'Successfully read {N} records')
    return sleep_groups


  def configure(self, config_string: str):
    raise NotImplementedError


  def report(self):
    console.show_info('UCDDB-1.0.0 Dataset')
    console.supplement(f'Totally {len(self.signal_groups)} subjects', level=2)

  # endregion: Abstract Methods (Data IO)

  # region: Overwriting

  def _check_data(self):
    """This method will be called during splitting dataset"""
    assert len(self.signal_groups) > 0

  # endregion: Overwriting

  # region: Data Visualization

  def show(self, channels: List[str] = None, **kwargs):
    from pictor import Pictor
    from pictor.plotters import Monitor

    # Initialize pictor and set objects
    p = Pictor(title='UCDDB-1.0.0', figure_size=(12, 8))
    p.objects = self.signal_groups

    # Set monitor
    channels = ['Lefteye', 'C3A2', 'ECG', 'Sound']
    m: Monitor = p.add_plotter(Monitor(channels=','.join(channels)))
    m.channel_list = [c for c, _, _ in self.signal_groups[0].name_tick_data_list]

    # .. set annotation logic
    anno_key = 'annotation'
    m.set(anno_key, self.STAGE_KEY)
    def on_press_a():
      if m.get(anno_key) is None: m.set(anno_key, self.STAGE_KEY)
      else: m.set(anno_key)
    m.register_a_shortcut('a', on_press_a, 'Flip annotation')

    # import matplotlib.style as mplstyle
    # mplstyle.use('fast')

    # Show pictor
    p.show()

  # endregion: Data Visualization



if __name__ == '__main__':
  from slp_core import th
  from slp.slp_agent import SLPAgent

  th.data_config = 'ucddb'

  # _ = UCDDB.load_raw_data(th.data_dir, save_xai_rec=True, overwrite=False)

  data_set = UCDDB.load_as_tframe_data(th.data_dir)
  data_set.report()
  data_set.show()

  # train_set, val_set, test_set = SLPAgent.load(th.data_dir)
  # assert isinstance(train_set, UCDDB)
  #
  # train_set.report()
  # train_set.show()

