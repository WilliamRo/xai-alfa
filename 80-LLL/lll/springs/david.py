from collections import OrderedDict

import numpy as np

from tframe.advanced.synapspring.springs.spring_base import SpringBase
from tframe import context
from tframe import hub as th
from tframe import tf
from tframe.utils.maths.stat_tools import Statistic



class David(SpringBase):

  name = 'CL-REG-DA'

  def __init__(self, model):
    # Call parent's initializer
    super(David, self).__init__(model)

  # region: Properties

  # endregion: Properties

  # region: Implementation of Abstract Methods

  def calculate_loss(self) -> tf.Tensor:
    vars = self.model.var_list
    shadows = self.model.shadows
    assert len(vars) == len(shadows)

    loss_list = []
    for v in vars:
      s = shadows[v]
      omega = self.omegas[v]
      loss_list.append(tf.reduce_mean(omega * tf.abs(s - v)))

    return tf.multiply(th.cl_reg_lambda, tf.add_n(loss_list), name=self.name)

  def _update_omega(self):
    ops = []
    for v in self.variables:
      ops.append(tf.assign(self.omegas[v], tf.abs(v)))
    self.model.session.run(ops)

  # endregion: Implementation of Abstract Methods

  # region: Exporting Tensors

  # endregion: Exporting Tensors


context.depot['david'] = David
context.depot['da'] = David







