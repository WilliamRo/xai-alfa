from tframe import mu
from tframe import Predictor
from tframe.nets.supotco import Supotco

import numpy as np



# Build model
model = Predictor(mark='Supotco')

model.add(mu.Input([32]))
model.add(mu.Dense(16, activation='relu'))

su: Supotco = model.add(Supotco())
li = su.init_a_limb('output-1')
li.add(mu.Dense(10, activation='softmax'))

li = su.init_a_limb('output-2')
li.add(mu.Dense(10, activation='softmax'))

su.set_gates([1, 0])

model.rehearse(export_graph=True)


# Predict
from tframe import DataSet
data = DataSet(np.zeros([1, 32]))
output = model.predict(data)
