from tframe.utils.note import Note
from pictor.pictor import Pictor, Plotter

import matplotlib.pyplot as plt
import numpy as np



# (1) Load data
note_path = r'03_fcnn/0406_fcnn.sum'
notes: list = Note.load(note_path)
code = '02'
note = [n for n in notes if n.configs['developer_code'] == code][0]
metric_key = 'Validati Accuracy'

# In data keys are names, values are list of numpy arrays
key = 'growth-record'
growth_record: dict = note.misc[key]
epoch_ticks = growth_record.pop('epoch_ticks')

# Pack objects
objects, labels = [], []
for label, stat_dict in growth_record.items():
  labels.append(label)
  objects.append({'/'.join(k.split('/')[1:3]): np.array(v)
                  for k, v in stat_dict.items()})


# (2) Visualize curves using Pictor
class GCVisualizer(Plotter):

  def __init__(self, pictor=None):
    # Call parent's constructor
    super(GCVisualizer, self).__init__(self.show_curves, pictor)

    self.new_settable_attr('show_metric', False, bool,
                           'Option to show metric curve')
    self.new_settable_attr('global_scale', False, bool,
                           'Option to use global scale')
    self.new_settable_attr('gs_percentile', 99.0, float,
                           'Percentile of global scale')
    self.new_settable_attr('xlim', None, str, 'xlim')

    self.register_a_shortcut('m', lambda: self.flip('show_metric'),
                             'Toggle show_metric')
    self.register_a_shortcut('g', lambda: self.flip('global_scale'),
                             'Toggle global_scale')

  # region: Properties

  @property
  def xlim(self):
    v = self.get('xlim')
    if not v: v = ','
    xmin, xmax = v.split(',')
    return [None if xmx == '' else float(xmx) for xmx in (xmin, xmax)]

  # endregion: Properties

  # region: APIs

  def set_global_scale_percentile(self, v: float):
    assert 0 <= v <= 100
    self.set('gs_percentile', v)
  sgp = set_global_scale_percentile

  # endregion: APIs

  def show_curves(self, x, title, ax: plt.Axes):
    curves = x
    assert isinstance(curves, dict)

    N = len(curves)
    margin = 0.05
    global_bound = None
    if self.get('global_scale'):
      p = self.get('gs_percentile')
      k = f'{title}-gs{p:.1f}'

      def _init(v_dict: dict):
        values = np.concatenate([np.ravel(v) for v in v_dict.values()])
        return np.percentile(values, p)

      global_bound = self.get_from_pocket(k, initializer=lambda: _init(curves))

    for i, (k, y) in enumerate(curves.items()):
      y = y - min(y)
      deno = max(y) if global_bound is None else global_bound
      y = y / deno * (1.0 - 2 * margin) + margin
      y = y + N - 1 - i

      ax.plot(epoch_ticks, y, color='grey')

    # Set y_ticks
    ax.set_ylim([0, N])
    ax.set_yticks([N - i - 0.5 for i in range(N)])
    ax.set_yticklabels(list(curves.keys()))

    # Set x-axis style
    mi, ma = self.xlim
    if mi is None: mi = epoch_ticks[0]
    if ma is None: ma = epoch_ticks[-1]
    ax.set_xlim(mi, ma)
    ax.set_xlabel('Epoch')
    ax.set_title(title)

    # Show metric curve if necessary
    if not self.get('show_metric'): return
    ax: plt.Axes = ax.twinx()
    ax.plot(note.step_array, note.scalar_dict[metric_key])
    ax.set_ylabel(metric_key)

GCVisualizer.plot(objects, labels=labels, fig_size=(12, 8))
