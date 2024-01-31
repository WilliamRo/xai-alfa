from tframe import DataSet
from tframe import pedia
from tframe.data.images.mnist import MNIST



def load_data() -> DataSet:
  from mg_core import th

  dataset, = MNIST.load(
    th.data_dir, validate_size=0, test_size=0, one_hot=False)

  dataset.features = dataset.features / 255.

  # [IMPORTANT] Be consistent with GAN.D.Input definition
  dataset.data_dict[pedia.D_input] = dataset.features

  return dataset



def probe(trainer):
  from tframe.trainers.trainer import Trainer
  from tframe.utils import imtool
  from tframe import mu

  import os

  assert isinstance(trainer, Trainer)

  # Generate figures
  model: mu.GAN = trainer.model
  samples = model.generate(sample_num=16)

  # Save snapshot
  file_path = os.path.join(
    model.agent.ckpt_dir, f'round-{trainer.total_rounds:.1f}.png')
  fig = imtool.gan_grid_plot(samples)
  imtool.save_plt(fig, file_path)
  return f'Image saved to `{file_path}`'



if __name__ == '__main__':
  dataset = load_data()

  from pictor import Pictor
  p = Pictor.image_viewer('MNIST')
  p.objects = dataset.features
  p.labels = [str(i) for i in dataset.dense_labels]
  p.show()


