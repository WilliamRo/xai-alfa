import numpy as np
# Put this line before importing tensorflow to get rid of future warnings
from tframe import console
import tensorflow as tf


console.suppress_logging()
console.start('Quan mask-size')
tf.InteractiveSession()
# =============================================================================
# Put your codes below
# =============================================================================
im = np.zeros(shape=[10, 5, 5, 1])

im_plchd = tf.placeholder(dtype=tf.float32, shape=[None, None, None, None])
feed_dict = {im_plchd: im}

S = 2

shape_tensor =  tf.shape(im_plchd)
shape_tensor = tf.stack(
  [shape_tensor[0]]
  + [shape_tensor[i+1] // S for i in range(2)]
  + [shape_tensor[-1]])

# console.eval_show(shape_tensor, feed_dict=feed_dict)

random_tensor = tf.random.uniform(shape_tensor)
input_size =  tf.shape(im_plchd)
random_tensor = tf.image.resize_images(
  random_tensor, size=(input_size[1], input_size[2]))

mask = random_tensor >= 0.5
mask = tf.cast(mask, im_plchd.dtype)

sess = tf.get_default_session()
value = sess.run(mask, feed_dict=feed_dict)

# Print results
print(value[0, ..., 0])
print()
print(value[1, ..., 0])
# =============================================================================
# End of the script
# =============================================================================
console.end()
