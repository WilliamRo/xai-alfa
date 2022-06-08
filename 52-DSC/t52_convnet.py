import dsc_core as core
import dsc_mu as m

from tframe import console
from tframe import tf
from tframe.utils.misc import date_string
from tframe.utils.organizer.task_tools import update_job_dir


# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'convnet'
id = 1
def model():
  th = core.th
  model = m.get_container()
  for a in th.archi_string.split('-'):
    if a == 'm':
      model.add(m.mu.MaxPool1D(2, 2))
    else:
      filters = int(a)
      model.add(m.mu.HyperConv1D(filters, th.kernel_size,
                                 activation=th.activation))
  return m.finalize(model, flatten=True)


def main(_):
  console.start('{} on DSC task'.format(model_name.upper()))

  th = core.th
  th.rehearse = False
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.data_config = 'rml:10;ap'
  th.val_proportion = 0.2
  th.test_proportion = 0.2

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
  th.use_rnn = False
  th.model = model

  th.archi_string = '16-16-m-32-32-m-64'
  th.kernel_size = 3
  th.activation = 'relu'
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 1000
  th.batch_size = 64

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

