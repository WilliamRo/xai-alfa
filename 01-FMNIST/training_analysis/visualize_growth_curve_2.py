from tframe.utils.note import Note
from pictor.pictor import Pictor, Plotter

import matplotlib.pyplot as plt
import numpy as np



# (1) Load data
note_path = r'01_cnn/0405_cnn.sum'
note_path = r'02_resnet/0405_resnet.sum'
note = Note.load(note_path)[1]
metric_key = 'Validati Accuracy'

# In data keys are names, values are list of numpy arrays
theta: dict = note.misc['theta']
epoch = theta.pop('epoch')
curves = {}
for k, v in theta.items():
  key = k.split('/')[1]
  curves[key] = np.array(v)


# (2) Visualize curves using Pictor
class GCVisualizer(Plotter):

  def __init__(self, pictor=None):
    # Call parent's constructor
    super(GCVisualizer, self).__init__(self.show_curves, pictor)

    self.new_settable_attr('show_metric', False, bool,
                           'Option to show metric curve')
    self.new_settable_attr('xlim', None, str, 'xlim')
    self.register_a_shortcut('m', lambda: self.flip('show_metric'),
                             'Toggle show_metric')

  @property
  def xlim(self):
    v = self.get('xlim')
    if not v: v = ','
    xmin, xmax = v.split(',')
    return [None if xmx == '' else float(xmx) for xmx in (xmin, xmax)]

  def show_curves(self, x, ax: plt.Axes):
    curves = x
    assert isinstance(curves, dict)

    N = len(curves)
    margin = 0.05
    for i, (k, y) in enumerate(curves.items()):
      y = y - min(y)
      y = y / max(y) * (1.0 - 2 * margin) + margin
      y = y + N - 1 - i

      ax.plot(epoch, y, color='grey')

    # Set y_ticks
    ax.set_yticks([N - i - 0.5 for i in range(N)])
    ax.set_yticklabels(list(curves.keys()))

    # Set x-axis style
    mi, ma = self.xlim
    if mi is None: mi = epoch[0]
    if ma is None: ma = epoch[-1]
    ax.set_xlim(mi, ma)
    ax.set_xlabel('Epoch')

    # Show metric curve if necessary
    if not self.get('show_metric'): return
    ax: plt.Axes = ax.twinx()
    ax.plot(note.step_array, note.scalar_dict[metric_key])
    ax.set_ylabel(metric_key)


GCVisualizer.plot([curves], fig_size=(8, 6))
