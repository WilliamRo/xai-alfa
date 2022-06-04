from tframe import console
from tframe import tf


console.suppress_logging()
console.start('Customize Gradient')
tf.InteractiveSession()
# =============================================================================
# Put your codes below
# =============================================================================
x, y = tf.constant(7.), tf.constant(8.)

z = x * y
dzdx, dzdy = tf.gradients([z], [x, y])

console.eval_show(z)
console.eval_show(dzdx)
console.eval_show(dzdy)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
@tf.custom_gradient
def multiply(a, b):
  def grad(dy):
    return dy * b * 2, dy * a * 2
  return a * b, grad

q = multiply(x, y)
dqdx, dqdy = tf.gradients([q], [x, y])

console.split()
console.eval_show(q)
console.eval_show(dqdx)
console.eval_show(dqdy)
# =============================================================================
# End of the script
# =============================================================================
console.end()
