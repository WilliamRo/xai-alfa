import fm_core as core
import fm_mu as m

from tframe import console
from tframe import tf
from tframe.nets.classic.conv_nets.resnet import ResNet
from tframe.utils.misc import date_string
from tframe.utils.organizer.task_tools import update_job_dir


# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'resnet'
id = 3
def model():
  th = core.th
  model = m.get_container(flatten=False)

  ResNet([int(n) for n in th.archi_string.split('-')],
         in_channels=th.filters, kernel_size=th.kernel_size).add_to(model)

  # Add flatten layer
  return m.finalize(model)


def main(_):
  console.start('{} on CIFAR-10 task'.format(model_name.upper()))

  th = core.th
  th.rehearse = False
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.centralize_data = False

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

  th.archi_string = '3-3-3'
  th.kernel_size = 3
  th.filters = 16
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 1000
  th.batch_size = 128

  th.optimizer = 'adam'
  th.learning_rate = 0.0003

  th.train = True
  th.overwrite = True
  th.patience = 10
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

