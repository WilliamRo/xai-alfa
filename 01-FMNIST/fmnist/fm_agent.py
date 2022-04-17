from tframe import console
from tframe import pedia
from tframe.data.base_classes import ImageDataAgent
from fmnist.fm_set import FMSet
from tframe.utils import misc

import gzip
import numpy as np
import os



class FMNIST(ImageDataAgent):

  DATA_NAME = 'FMNIST'
  DATA_URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/'
  TFD_FILE_NAME = 'fmnist.tfd'

  PROPERTIES = {
    pedia.classes: ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
    FMSet.NUM_CLASSES: 10
  }


  @classmethod
  def load(cls, data_dir, train_size=-1, validate_size=5000, test_size=10000,
           flatten=False, one_hot=True, over_classes=False, **kwargs):
    data_set = cls.load_as_tframe_data(data_dir)
    data_set.features  = data_set.features / 255.0
    if flatten:
      data_set.features = data_set.features.reshape(data_set.size, -1)
    if one_hot:
      data_set.targets = misc.convert_to_one_hot(
        data_set.targets, data_set[data_set.NUM_CLASSES])

    return cls._split_and_return(
      data_set, train_size, validate_size, test_size, over_classes=over_classes)


  @classmethod
  def load_as_tframe_data(cls, data_dir):
    file_path = os.path.join(data_dir, cls.TFD_FILE_NAME)
    if os.path.exists(file_path): return FMSet.load(file_path)

    # If .tfd file does not exist, try to convert from raw data
    console.show_status('Trying to convert raw data to tframe DataSet ...')
    images, labels = cls.load_as_numpy_arrays(data_dir)
    data_set = FMSet(images, labels, name=cls.DATA_NAME, **cls.PROPERTIES)

    # Generate groups if necessary
    if data_set.num_classes is not None:
      groups = []
      dense_labels = misc.convert_to_dense_labels(labels)
      for i in range(data_set.num_classes):
        # Find samples of class i and append to groups
        samples = list(np.argwhere([j == i for j in dense_labels]).ravel())
        groups.append(samples)
      data_set.properties[data_set.GROUPS] = groups

    # Show status
    console.show_status('Successfully converted {} samples'.format(
      data_set.size))
    # Save DataSet
    console.show_status('Saving data set ...')
    data_set.save(file_path)
    console.show_status('Data set saved to {}'.format(file_path))
    return data_set


  @classmethod
  def load_as_numpy_arrays(cls, data_dir):
    # Define .gz file names
    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

    # Check gs files
    for file_name in (TRAIN_IMAGES, TRAIN_LABELS, TEST_IMAGES, TEST_LABELS):
      cls._check_raw_data(data_dir, file_name, cls.DATA_URL + file_name)

    # Extract images and labels
    get_file_path = lambda fn: os.path.join(data_dir, fn)
    # 60000 training samples
    train_images = cls._extract_file(get_file_path(TRAIN_IMAGES), 'image')
    train_labels = cls._extract_file(get_file_path(TRAIN_LABELS), 'label')
    # 10000 test samples
    test_images = cls._extract_file(get_file_path(TEST_IMAGES), 'image')
    test_lables = cls._extract_file(get_file_path(TEST_LABELS), 'label')
    # Merge data into regular numpy arrays
    images = np.concatenate((train_images, test_images))
    labels = np.concatenate((train_labels, test_lables))
    # Return data tuple
    return images, labels


  @classmethod
  def _extract_file(cls, path, image_or_label):
    assert image_or_label in ('image', 'label')
    offset = 16 if image_or_label == 'image' else 8
    with gzip.open(path, 'rb') as file:
      data = np.frombuffer(file.read(), np.uint8, offset=offset)
    if image_or_label == 'image':
      return  data.reshape(len(data)//(28*28), 28, 28, 1)
    return data



if __name__ == '__main__':
  from fm_core import th
  data_set = FMNIST.load_as_tframe_data(th.data_dir)

