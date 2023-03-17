# Put this line before importing tensorflow to get rid of future warnings
from tframe import console
from tframe import tf
from tframe import constraints



console.suppress_logging()
console.start('constraints')
tf.InteractiveSession()
# =============================================================================
# Put your codes below
# =============================================================================
constraint = constraints.get('value')
x = tf.get_variable('x', shape=[2, 2], dtype=tf.float32,
                    initializer=tf.zeros_initializer(),
                    constraint=constraint)
g = [[1.1, -0.8], [2.2, -3.1]]
optimizer = tf.train.GradientDescentOptimizer(1.0)
op = optimizer.apply_gradients([(g, x)])

sess = tf.get_default_session()
sess.run(tf.global_variables_initializer())

console.eval_show(x)
sess.run(op)
console.eval_show(x)

# =============================================================================
# End of the script
# =============================================================================
console.end()
