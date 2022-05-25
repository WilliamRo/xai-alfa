import numpy as np

from to_core import th
import to_du as du


# Load data
th.fixed_length = True
th.bits = 3
th.sequence_length = 30
_, val_set, _ = du.load(
  th.data_dir, th.sequence_length, th.bits, th.fixed_length, th.val_size,
  th.test_size)


# Retrieve strings
alphabet = ['X', 'Y', 'a', 'b', 'c', 'd']
sequences = [''.join([alphabet[index] for index in np.argmax(seq, axis=-1)])
             for seq in val_set.features]


# -----------------------------------------------------------------------------
#  Visualize sequences
# -----------------------------------------------------------------------------
from pictor import DaVinci
import matplotlib.pyplot as plt


da = DaVinci(title='Temporal Order', height=1, width=1)
da.objects = sequences

def get_show_text_method(index):
  def _show_text(x: str, ax: plt.Axes):
    ax.cla()
    ax.text(0.5, 0.5, x[index], ha='center', va='center', size=50)
    ax.set_axis_off()
  return _show_text

for i in range(len(sequences[0])): da.add_plotter(get_show_text_method(i))

da.show()