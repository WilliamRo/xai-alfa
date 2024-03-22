from collections import OrderedDict
from tframe import DataSet
from tframe import pedia
from tframe.data.images.mnist import MNIST



def load_data() -> DataSet:
  from md_core import th

  dataset, = MNIST.load(
    th.data_dir, validate_size=0, test_size=0, one_hot=False)

  dataset.features = dataset.features / 255.

  return dataset



def probe(trainer):
  from tframe.trainers.trainer import Trainer
  from tframe.utils import imtool
  from tframe import mu
  from tframe import hub as th

  import os
  import numpy as np

  assert isinstance(trainer, Trainer)

  # Inline settings
  FIX_X_T = True
  SAMPLE_NUM = 16

  # Generate figures
  x_T = None
  if FIX_X_T: x_T = th.get_from_pocket(
    'DDPM_X_T_PROBE', initializer=lambda: np.random.randn(
      SAMPLE_NUM, *th.input_shape))

  model: mu.GaussianDiffusion = trainer.model
  samples = model.generate(sample_num=SAMPLE_NUM, x_T=x_T)

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

  # Early stop
  REC_KEY = 'REC_KEY'
  best_loss, patience = trainer.get_from_pocket(
    REC_KEY, initializer=lambda: (1e10, 0))
  loss = scalars['Loss']
  if loss < best_loss:
    best_loss, patience = loss, 0
    trainer._save_model()
  else:
    patience += 1
    if not th.early_stop: trainer._save_model()   # TODO: save for no reason
    if patience > th.patience: trainer.th.raise_stop_flag()
  trainer.put_into_pocket(REC_KEY, (best_loss, patience), exclusive=False)

  return f'[{patience}/{th.patience}] Image saved to `{file_path}`'



if __name__ == '__main__':
  dataset = load_data()

  from pictor import Pictor
  p = Pictor.image_viewer('MNIST')
  p.objects = dataset.features
  p.labels = [str(i) for i in dataset.dense_labels]
  p.show()


