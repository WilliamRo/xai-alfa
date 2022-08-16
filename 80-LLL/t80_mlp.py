import lll_core as core
import lll_mu as m

from tframe.enums import SaveMode
from tframe import console
from tframe import tf
from tframe.utils.misc import date_string
from tframe.utils.organizer.task_tools import update_job_dir


# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'mlp'
id = 8
def model():
  th = core.th
  model = m.get_container(flatten=True)
  for n in core.th.archi_string.split('-'):
    model.add(m.mu.Dense(int(n), activation=th.activation))
    if th.dropout > 0: model.add(m.mu.Dropout(1 - th.dropout))
  return m.finalize(model)


def main(_):
  console.start('{} on LLL task'.format(model_name.upper()))

  th = core.th
  th.rehearse = False
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.task = th.Tasks.FMNIST

  th.data_config = '2,1,1,1'
  th.train_id = 0
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

  th.archi_string = '128'
  th.activation = 'relu'
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 10000
  th.batch_size = 128

  th.optimizer = 'adam'
  th.learning_rate = 0.0003
  th.patience = 5

  th.validation_per_round = 3

  th.train = True
  th.overwrite = True if th.train_id == 0 else False
  th.save_mode = SaveMode.ON_RECORD
  th.print_cycle = 5
  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = '{}({})'.format(model_name, th.archi_string)
  th.gather_summ_name = th.prefix + summ_name + '.sum'
  core.activate()


if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()

