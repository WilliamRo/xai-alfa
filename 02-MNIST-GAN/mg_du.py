from collections import OrderedDict
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
  from tframe import hub as th

  import os

  assert isinstance(trainer, Trainer)

  # Inline settings
  FIX_Z = True
  SAMPLE_NUM = 16

  # Generate figures
  z = None
  if FIX_Z: z = th.get_from_pocket(
    'GAN_Z_PROBE', initializer=lambda: trainer.model._random_z(SAMPLE_NUM))

  model: mu.GAN = trainer.model
  samples = model.generate(sample_num=SAMPLE_NUM, z=z)

  # Save snapshot
  file_path = os.path.join(
    model.agent.ckpt_dir, f'round-{trainer.total_rounds:.1f}.png')
  fig = imtool.gan_grid_plot(samples)
  imtool.save_plt(fig, file_path)

  # Take notes for export
  scalars = OrderedDict(
    {k.name: v.running_average for k, v in trainer.batch_loss_stats.items()})
  trainer.model.agent.take_down_scalars_and_tensors(scalars, OrderedDict())
  trainer._inter_cut('Notes taken down.', prompt='[Export]')

  return f'Image saved to `{file_path}`'



if __name__ == '__main__':
  dataset = load_data()

  from pictor import Pictor
  p = Pictor.image_viewer('MNIST')
  p.objects = dataset.features
  p.labels = [str(i) for i in dataset.dense_labels]
  p.show()


