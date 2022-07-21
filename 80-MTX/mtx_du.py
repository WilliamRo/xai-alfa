from tframe import DataSet
from mtx.mtx_agent import MTXAgent

import os



def load_data():

  # Load data
  train_set, val_set, test_set = MTXAgent.load()

  assert isinstance(train_set, DataSet)
  assert isinstance(val_set, DataSet)
  assert isinstance(test_set, DataSet)

  return train_set, val_set, test_set



if __name__ == '__main__':
  train_set, val_set, test_set = load_data()
  # train_set.show()

  from pictor import Pictor
  p = Pictor.image_viewer('FMNIST')
  p.objects = train_set.features
  p.labels = [train_set['CLASSES'][i] for i in train_set.dense_labels]
  p.show()


