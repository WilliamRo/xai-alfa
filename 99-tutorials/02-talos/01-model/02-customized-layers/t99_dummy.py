import dsc_core as core
import dsc_mu as m

from tframe import console
from tframe import tf
from tframe.utils.misc import date_string
from tframe.utils.organizer.task_tools import update_job_dir

from dsc.dsc_layers.dummy import Dummy


# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'dummy'
id = 9
def model():
  th = core.th
  model = m.get_container()
  model.add(m.mu.Flatten())
  model.add(Dummy(dim=88))
  return m.finalize(model)


def main(_):
  console.start('{} on DSC task'.format(model_name.upper()))

  th = core.th
  th.rehearse = True
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.data_config = 'rml:10;iq'
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

  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 1000
  th.batch_size = 64

  th.optimizer = 'adam'
  th.learning_rate = 0.0003

  th.train = False
  th.save_model = False
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

