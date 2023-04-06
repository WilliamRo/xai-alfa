from tframe.trainers.trainer import Trainer
from tframe.utils.maths.stat_tools import Statistic
from tframe import context

import numpy as np



def put_statistics_to_note(trainer: Trainer):
  """History of weights and gradients are stored inside
     (1) monitor._weight_history and
     (2) monitor._weight_grad_dict, respectively.
  """
  note, monitor = context.note, context.monitor

  # Initialize `theta` dict if necessary
  key = 'growth-record'
  stats_labels = ['Weight-Stats', 'Grad-Stats']
  if key not in note.misc:
    note.misc[key] = {k: {w.name: [] for w in monitor._weights_list}
                      for k in stats_labels}
    note.misc[key]['epoch_ticks'] = []

  growth_record = note.misc[key]
  # Append data into lists in note.misc['theta']
  growth_record['epoch_ticks'].append(trainer.total_rounds)

  # Takedown weight-related statistics
  d = growth_record[stats_labels[0]]
  for w, s in monitor._weight_history.items():
    assert isinstance(s, Statistic)
    v = np.linalg.norm(s._value_list[-2] - s.last_value)
    d[w.name].append(v)

  # Takedown grad-related statistics
  d = growth_record[stats_labels[1]]
  for w, s, in monitor._weight_grad_dict.items():
    assert isinstance(s, Statistic)
    v = np.linalg.norm(s.last_value)
    d[w.name].append(v)

