from collections import OrderedDict
from pictor import Pictor
from tframe.utils.note import Note
from pictor.plotters.plotter_base import Plotter

import os
import numpy as np
from roma import console
import matplotlib.pyplot as plt




class Bamboo(Plotter):

  N_SPLITS = 5
  GROUP_KEYS = ('cl_reg_config', 'cl_reg_lambda', 'trial_id', 'developer_code',
                'balance_classes')

  def __init__(self, pictor=None):
    super(Bamboo, self).__init__(self.draw_bamboo, pictor)

    self.summ_path = None
    self.new_settable_attr('title', True, bool, 'Option to show title')
    self.new_settable_attr('x0', 0, float, 'x-axis start point')
    self.new_settable_attr('star', False, bool, 'Option to show star')


  def draw_bamboo(self, x: list, ax: plt.Axes):
    notes = x
    n_splits = self.N_SPLITS

    patience = notes[0].configs['patience']
    k = patience * 2

    star = self.get('star')
    # ----------------------------------------------------------------------
    #  Retrieve package
    # ----------------------------------------------------------------------
    acc_keys = [f'Test-{i + 1} Accuracy' for i in range(n_splits)]
    if star: acc_keys.append('Test-* Accuracy')

    package = [[n.step_array] + [n.scalar_dict[k] for k in acc_keys] for n in
               notes]
    id = notes[0].configs['trial_id']

    # Show average accuracy
    index = np.argmax(notes[-1].scalar_dict['Val-5 Accuracy'])
    avg_acc = np.average([array[index] for array in package[-1][1:n_splits+1]])
    # ----------------------------------------------------------------------
    #  Draw figure
    # ----------------------------------------------------------------------
    colors = ['tab:red', 'tab:orange', 'gold', 'tab:green', 'tab:cyan',
              'tab:blue', 'tab:purple']

    y_min = min([min(np.concatenate([a for i, a in enumerate(arrays) if i > 0]))
                 for arrays in package])
    y_max = max([max(np.concatenate([a for i, a in enumerate(arrays) if i > 0]))
                 for arrays in package])

    end_points = [(0, 0) for _ in range(n_splits + star)]
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

        label = None
        if i == j: label = f'Task-{i+1}'
        if i >= n_splits and j == len(package) - 1: label = f'Task-*'

        ax.plot(x[:_k], acc[:_k], color=colors[i], linewidth=width, alpha=alpha,
                label=label)

        # Record endpoints
        end_points[i] = (x[_k - 1], acc[_k - 1])

    # Set style
    # ax.legend([f'Split-{i+1}' for i in range(n_splits)])
    ax.legend()
    ax.set_xlim([self.get('x0'), None])

    if self.get('title'):
      config, lambd, code, bc = [
        self.get_config(k, notes[0].configs) for k in (
          'cl_reg_config', 'cl_reg_lambda', 'developer_code',
          'balance_classes')]
      title = f'[T-{id}]'
      title += f' {config} ($\lambda$={lambd})'
      title += f', avg(acc)={avg_acc * 100:.2f}'
      title += f', {"n" if not bc else ""}bc'
      title += f' - {code}'
      ax.set_title(title)

  # region: Public Methods

  def load_notes(self, summ_path=None):
    assert summ_path is not None

    notes = Note.load(summ_path)
    console.show_status(f'{len(notes)} found.')

    note_groups = []

    od = OrderedDict()
    for n in notes:
      od[tuple([self.get_config(k, n.configs) for k in self.GROUP_KEYS])] = None
    group_keys = list(od.keys())

    for cfgs in group_keys:
      _notes = [n for n in notes if all(
        [self.get_config(k, n.configs) == cfg
         for cfg, k in zip(cfgs, self.GROUP_KEYS)])]
      if len(_notes) == self.N_SPLITS: note_groups.append(_notes)

    assert len(note_groups) > 0
    self.summ_path = summ_path

    self.pictor.static_title = f'Bamboo - {os.path.basename(summ_path)}'

    self.pictor.objects = note_groups[::-1]
    self.pictor.set_object_cursor(1)
    self.pictor.refresh()

  @staticmethod
  def get_config(key, configs):
    assert isinstance(configs, dict)
    if key in configs: return configs[key]
    return '-'

  # endregion: Public Methods

  # region: Commands

  def register_shortcuts(self):
    self.register_a_shortcut('r', lambda: self.load_notes(self.summ_path),
                             description='Reload notes')
    self.register_a_shortcut('t', lambda: self.flip('title'), 'Toggle title')
    self.register_a_shortcut('s', lambda: self.flip('star'), 'Toggle star')

  # endregion: Commands



if __name__ == '__main__':
  # summ_path = r'E:\xai-alfa\80-LLL\xmnist\08_mlp\0906_s80_mlp.sum'
  summ_path = r'E:\xai-alfa\80-LLL\xmnist\09_cnn\0908_s80_cnn.sum'
  # summ_path = r'E:\xai-alfa\80-LLL\xmnist\01_matt\0907_s80_matt.sum'
  # summ_path = r'E:\xai-alfa\80-LLL\xmnist\02_mark\0908_s80_mark.sum'

  p = Pictor(figure_size=(10, 5))
  bb = Bamboo(p)
  plotter = p.add_plotter(bb)
  bb.load_notes(summ_path)
  p.show()
