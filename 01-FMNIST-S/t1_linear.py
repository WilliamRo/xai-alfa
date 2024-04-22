import fm_core as core
import fm_mu as m

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
    # model.add(m.mu.Dense(int(n), activation=th.activation))
    model.add(m.mu.Dense(int(n)))
    if th.dropout > 0: model.add(m.mu.Dropout(1 - th.dropout))
  return m.finalize(model)


def main(_):
  console.start('{} on FMNIST task'.format(model_name.upper()))

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

  th.archi_string = '128'
  th.activation = 'relu'
  th.dropout = 0.0
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 1
  th.batch_size = 128

  th.optimizer = 'adam'
  th.learning_rate = 0.0003

  th.train = 1
  th.overwrite = True
  th.print_cycle = 20
  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = '{}({})_{}'.format(model_name, th.archi_string, th.activation)
  th.gather_summ_name = th.prefix + summ_name + '.sum'
  core.activate()


if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()

