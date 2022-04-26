import fm_core as core
import fm_mu as m

from tframe import console
from tframe import tf
from tframe.utils.misc import date_string
from tframe.utils.organizer.task_tools import update_job_dir


# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'cnn'
id = 2
def model():
  th = core.th
  model = m.get_container(flatten=False)

  for i, c in enumerate(core.th.archi_string.split('-')):
    if c == 'p':
      model.add(m.mu.MaxPool2D(pool_size=2, strides=1))
      continue

    c = int(c)
    model.add(m.mu.Conv2D(
      filters=c, kernel_size=th.kernel_size,
      activation=th.activation, use_batchnorm=th.use_batchnorm and i > 0))

  # Add flatten layer
  model.add(m.mu.Flatten())
  return m.finalize(model)


def main(_):
  console.start('{} on CIFAR-10 task'.format(model_name.upper()))

  th = core.th
  th.rehearse = False
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  pass

  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  update_job_dir(id, model_name)
  summ_name = model_name
  th.prefix = '{}_'.format(date_string())

  th.visible_gpu_id = 0
  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model

  th.archi_string = '64-p-32'
  th.kernel_size = 3
  th.activation = 'relu'
  th.use_batchnorm = False
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 1000
  th.batch_size = 128

  th.optimizer = 'adam'
  th.learning_rate = 0.0003

  th.train = True
  th.overwrite = True
  th.print_cycle = 20
  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = '{}({})'.format(model_name, th.archi_string)
  th.gather_summ_name = th.prefix + summ_name + '.sum'
  core.activate()


if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()

