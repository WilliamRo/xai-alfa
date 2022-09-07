from collections import OrderedDict

import numpy as np

from tframe.advanced.synapspring.springs.spring_base import SpringBase
from tframe import context
from tframe import hub as th
from tframe import tf
from tframe.utils.maths.stat_tools import Statistic



class Matt(SpringBase):

  name = 'CL-REG-MA'

  def __init__(self, model):
    # Call parent's initializer
    super(Matt, self).__init__(model)

  # region: Properties

  # endregion: Properties

  # region: Implementation of Abstract Methods

  def _update_omega(self):
    ops = []
    for i, v in enumerate(self.variables):
      if i != 0: continue
      shape = v.shape.as_list()
      ops.append(tf.assign(self.omegas[v], 1e6 * np.ones(shape)))
    self.model.session.run(ops)

  # endregion: Implementation of Abstract Methods

  # region: Exporting Tensors

  # endregion: Exporting Tensors


context.depot['matt'] = Matt
context.depot['ma'] = Matt







