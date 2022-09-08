import numpy as np
# Put this line before importing tensorflow to get rid of future warnings
from tframe import console
from tframe import mu
from tframe import tf


console.suppress_logging()
console.start('Flatten')
tf.InteractiveSession()
# =============================================================================
# Put your codes below
# =============================================================================
x = np.arange(12).reshape(1, 2, 6)
print(x)

tensor_x = tf.constant(x)
console.eval_show(tensor_x)

flattened_x = mu.Flatten()(tensor_x)
console.eval_show(flattened_x)
# =============================================================================
# End of the script
# =============================================================================
console.end()
