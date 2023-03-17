import numpy as np
# Put this line before importing tensorflow to get rid of future warnings
from tframe import console
from tframe import tf


console.suppress_logging()
console.start('sign(W)')
tf.InteractiveSession()
# =============================================================================
# Put your codes below
# =============================================================================
from tframe.operators.kernel_base import KernelBase

W = tf.constant([[-1.2, 2.3], [0.2, 5]], dtype=tf.float32)
x = tf.constant([[1.0], [-1.0]], dtype=tf.float32)

y = W @ x
L = tf.norm(y)

y_b = KernelBase.binarize_weights(W) @ x
L_b = tf.reduce_sum(y_b)

dLdW = tf.gradients(L, W)[0]
dLdW_b = tf.gradients(L_b, W)[0]

console.eval_show(W)
console.eval_show(x)
# console.eval_show(y)
# console.eval_show(L)
# console.eval_show(dLdW)
console.eval_show(dLdW_b)

# =============================================================================
# End of the script
# =============================================================================
console.end()
