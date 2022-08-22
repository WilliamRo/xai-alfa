import numpy as np

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
summ_path = r'E:\xai-alfa\80-LLL\xmnist\09_cnn\0816_s80_cnn.sum'
trial_id = 1

notes = Note.load(summ_path)
notes = [n for n in notes if n.configs['trial_id'] == trial_id]

n_splits = len(notes[0].configs['data_config'].split(','))
assert len(notes) == n_splits
k = notes[0].configs['patience'] * 2
# ----------------------------------------------------------------------
#  Retrieve package
# ----------------------------------------------------------------------
acc_keys = [f'Train-{i+1} Accuracy' for i in range(4)]

package = [[n.step_array] + [n.scalar_dict[k] for k in acc_keys] for n in notes]
# ----------------------------------------------------------------------
#  Show in Pictor
# ----------------------------------------------------------------------
colors = ['tab:red', 'tab:orange', 'gold', 'tab:green', 'tab:cyan', 'tab:blue',
          'tab:purple']
def plotter(ax: plt.Axes):
  y_min = min([min(np.concatenate([a for i, a in enumerate(arrays) if i > 0]))
               for arrays in package])
  y_max = max([max(np.concatenate([a for i, a in enumerate(arrays) if i > 0]))
               for arrays in package])

  end_points = [(0, 0) for _ in range(n_splits)]
  for j, arrays in enumerate(package):
    x = arrays.pop(0)
    if j == n_splits - 1: _k = -k
    else:
      next_x = package[j + 1][0]
      _k = max(np.argwhere(x < next_x[0]))[0] + 1

    # Draw vertical lines
    if j > 0: ax.plot([end_points[0][0], end_points[0][0]],
                      [y_min, y_max], color='#ccc')

    for i, acc in enumerate(arrays):
      # Draw dashed lines
      if j > 0: ax.plot([end_points[i][0], x[0]], [end_points[i][1], acc[0]],
                        ':', color=colors[i])

      # Draw acc curve
      width = 2 if i == j else 1
      alpha = 1 if i == j else 0.7
      ax.plot(x[:_k], acc[:_k], color=colors[i], linewidth=width, alpha=alpha)

      # Record endpoints
      end_points[i] = (x[_k - 1], acc[_k - 1])

  # Set style
  ax.legend([f'Split-{i+1}' for i in range(n_splits)])
  ax.set_xlim([2, None])

p = Pictor(figure_size=(10, 5))
plotter = p.add_plotter(plotter)
p.show()
