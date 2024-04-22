from tframe import mu
from tframe import Predictor

import os



# -----------------------------------------------------------------------------
# Initialize a tframe.Predictior
# -----------------------------------------------------------------------------
model = Predictor(mark='gnet1')
model.add(mu.Input(sample_shape=[28, 28, 1]))

# -----------------------------------------------------------------------------
# Add a DAG block
# -----------------------------------------------------------------------------
# Put your code here
n = 8
vertices = [
  [mu.Conv2D(n, kernel_size=1, activation='relu'),
   mu.Conv2D(n, kernel_size=1, activation='relu')],
  mu.Conv2D(n, kernel_size=3, activation='relu'),
  mu.Conv2D(n, kernel_size=5, activation='relu'),
  mu.MaxPool2D(pool_size=3, strides=1),
  mu.Merge.Concat(axis=-1)]
edges = '1;10;100;1000;01111'

fm = mu.ForkMergeDAG(vertices, edges)

model.add(fm)
# -----------------------------------------------------------------------------
# Add output layer and rehearse
# -----------------------------------------------------------------------------
model.add(mu.Flatten())
model.add(mu.Dense(num_neurons=10, activation='softmax'))
model.rehearse(
  export_graph=True, build_model=True, path=os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'dag'))
