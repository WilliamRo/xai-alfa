from pictor import Pictor
from tframe.utils.note import Note

import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
#  Read notes
# ----------------------------------------------------------------------
"""
note.configs: dict
note.scalar_dict: {key: scalar_array}
note.step_array: numpy array

"""
summ_path = r'E:\xai-alfa\80-LLL\09_cnn\0812_s80_cnn.sum'
notes = Note.load(summ_path)

# ----------------------------------------------------------------------
#  Read notes
# ----------------------------------------------------------------------
acc_keys = [f'Train-{i+1} Accuracy' for i in range(4)]
package = [[n.step_array] + [n.scalar_dict[k] for k in acc_keys] for n in notes]

# ----------------------------------------------------------------------
#  Show in Pictor
# ----------------------------------------------------------------------
colors = ['tab:red', 'tab:orange', 'gold', 'tab:green', 'tab:cyan', 'tab:blue',
          'tab:purple']
def plotter(ax: plt.Axes):
  for arrays in package:
    x = arrays.pop(0)
    for i, acc in enumerate(arrays):
      ax.plot(x, acc, color=colors[i])

  # Set style
  # ax.set_ylim([0.7, 1.0])

p = Pictor(figure_size=(10, 5))
p.add_plotter(plotter)
p.show()
