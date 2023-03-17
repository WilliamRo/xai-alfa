import numpy as np
# Put this line before importing tensorflow to get rid of future warnings
from tframe import console
from tframe import tf


console.suppress_logging()
console.start('sign_activation')
tf.InteractiveSession()
# =============================================================================
# Put your codes below
# =============================================================================
@tf.custom_gradient
def sign_st(x):
  """Sign function with straight-through estimator"""
  def sign(v):
    return (tf.cast(tf.math.greater_equal(v, 0), tf.float32) - 0.5) * 2

  def grad(dy):
    return dy * tf.cast(tf.logical_and(
      tf.greater_equal(x, -1.0), tf.less_equal(x, 1.0)), dtype=tf.float32)
  return sign(x), grad


x = tf.constant([-0.2, 0, 2.3], dtype=tf.float32)
y = sign_st(x)
L = tf.reduce_sum(y)
dLdx = tf.gradients(L, x)[0]

console.eval_show(y)
console.eval_show(dLdx)

# =============================================================================
# End of the script
# =============================================================================
console.end()
