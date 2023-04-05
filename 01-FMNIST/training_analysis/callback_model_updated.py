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
  key = 'theta'
  if key not in note.misc:
    note.misc[key] = {w.name: [] for w in monitor._weights_list}
    note.misc[key]['epoch'] = []

  d = note.misc[key]
  # Append data into lists in note.misc['theta']
  d['epoch'].append(trainer.total_rounds)

  # Takedown weight-related statistics
  for w, s in monitor._weight_history.items():
    assert isinstance(s, Statistic)
    v = np.linalg.norm(s._value_list[0] - s.last_value)
    d[w.name].append(v)

  # Takedown grad-related statistics
