from tframe import tf
from tframe.layers.layer import Layer
from tframe.layers.layer import single_input



class Dummy(Layer):
  abbreviation = 'dummy'
  full_name = abbreviation

  def __init__(self, dim=88):
    self.dim = dim

  @property
  def structure_tail(self): return f'({self.dim})'

  @single_input
  def _link(self, x: tf.Tensor):

    W = tf.get_variable(
      'W', shape=[x.shape[1], self.dim], dtype=tf.float32,
      initializer=tf.initializers.glorot_normal)
    y = tf.matmul(x, tf.exp(W))
    return y



