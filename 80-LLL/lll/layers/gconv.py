from tframe import tf
from tframe.layers.hyper.conv import Conv2D as HyperConv2D
from tframe import hub as th

import typing as tp



class GatedConv2D(HyperConv2D):

  full_name = 'gconv2d'
  abbreviation = 'gconv2d'

  def __init__(self,
               filters,
               kernel_size,
               strides=1,
               padding='same',
               dilations=1,
               activation=None,
               use_bias=False,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               expand_last_dim=False,
               use_batchnorm=False,
               filter_generator=None,
               name: tp.Optional[str] = None,
               **kwargs):

    # Call parent's constructor
    super(GatedConv2D, self).__init__(
      filters, kernel_size, strides, padding, dilations, activation,
      use_bias, kernel_initializer, bias_initializer, expand_last_dim,
      use_batchnorm, filter_generator, name, **kwargs)


  def get_layer_string(self, scale, full_name=False, suffix=''):
    activation = self._activation_string
    if self.dilations not in (None, 1): suffix += f'(di{self.dilations})'
    suffix += f'<gs{th.group_size}>'
    if callable(self.filter_generator): suffix += '[H]'
    if self.use_batchnorm: suffix += '->bn'
    if isinstance(activation, str): suffix += '->{}'.format(activation)
    result = super().get_layer_string(scale, full_name, suffix)
    return result


  def forward(self, x: tf.Tensor, filter=None, **kwargs):
    from tframe import mu

    # Calculate groups
    c = self.channels * th.group_size
    groups = self.conv2d(
      x, c, self.kernel_size, 'GroupedConv2D',
      strides=self.strides, padding=self.padding,
      dilations=self.dilations, filter=filter, **kwargs)

    shape = groups.shape.as_list()
    shape = shape[:-1] + [self.channels, th.group_size]
    groups = tf.reshape(groups, shape=[-1] + shape[1:])

    # Calculate group gates
    gates = self.conv2d(
      x, c, self.kernel_size, 'GroupGates',
      strides=self.strides, padding=self.padding,
      dilations=self.dilations, filter=filter, **kwargs)

    gates = mu.GlobalAveragePooling2D()(gates)
    gates = tf.reshape(gates, [-1, 1, 1, self.channels, th.group_size])
    gates = tf.nn.softmax(gates)

    # Apply gates and return
    y = groups * gates
    y = tf.reduce_sum(y, axis=-1)

    return y
