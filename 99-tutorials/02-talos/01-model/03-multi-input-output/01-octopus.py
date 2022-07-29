from tframe import mu
from tframe import Predictor
from tframe.nets.octopus import Octopus

import numpy as np



# Build model
model = Predictor(mark='octopus-5')

oc: Octopus = model.add(Octopus())

li = oc.init_a_limb('input-1', [32])
li.add(mu.Dense(8))

li = oc.init_a_limb('input-2', [16])
li.add(mu.Dense(8))

oc.set_gates([1, 0])

model.add(mu.Activation('relu'))
model.add(mu.Dense(10, activation='softmax'))
model.rehearse(export_graph=True)


# Predict
from tframe import DataSet
data = DataSet(data_dict={'input-1': np.zeros([1, 32])})
output = model.predict(data)










