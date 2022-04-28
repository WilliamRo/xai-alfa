from fmnist.fm_agent import FMNIST
from tframe import DataSet
from tframe.data.augment.img_aug import image_augmentation_processor



def load_data():
  from fm_core import th
  train_set, val_set, test_set = FMNIST.load(th.data_dir)
  assert isinstance(train_set, DataSet)
  assert isinstance(val_set, DataSet)
  assert isinstance(test_set, DataSet)
  # Set batch_preprocessor for augmentation if required
  if th.augmentation:
    train_set.batch_preprocessor = image_augmentation_processor
  return train_set, val_set, test_set



if __name__ == '__main__':
  train_set, val_set, test_set = load_data()
  train_set.show()

