from tframe import mu
from tframe import Predictor

import os



# -----------------------------------------------------------------------------
# Initialize a tframe.Predictior
# -----------------------------------------------------------------------------
model = Predictor(mark='dag')
model.add(mu.Input(sample_shape=[32, 32, 3]))
h1 = model.add(mu.Conv2D(128, kernel_size=3))

# -----------------------------------------------------------------------------
# Add a DAG block
# -----------------------------------------------------------------------------
# Put your code here


# -----------------------------------------------------------------------------
# Add output layer and rehearse
# -----------------------------------------------------------------------------
model.add(mu.GlobalAveragePooling2D())
model.add(mu.Flatten())
model.add(mu.Dense(num_neurons=10, activation='softmax'))
model.rehearse(
  export_graph=True, build_model=True, path=os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'dag'))
