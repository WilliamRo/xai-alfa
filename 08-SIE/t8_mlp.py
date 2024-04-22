import sie_core as core
import sie_mu as m

from tframe import console
from tframe import tf
from tframe.utils.misc import date_string
from tframe.utils.organizer.task_tools import update_job_dir


# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'mlp'
id = 1
def model():
  th = core.th
  model = m.get_container()
  if th.archi_string != '':
    for n in th.archi_string.split('-'):
      model.add(m.mu.Dense(int(n), activation=th.activation))
  return m.finalize(model)


def main(_):
  th = core.th
  console.start('{} on {}'.format(model_name.upper(), th.data_config))
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.data_config = [
    'gordon-424',    # 0
  ][0]

  th.folds_i = 1
  th.folds_k = 5
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

  th.archi_string = '1'
  th.activation = 'relu'
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 1000
  th.batch_size = 32
  th.updates_per_round = 50

  th.optimizer = 'adam'
  th.learning_rate = 0.0003

  th.train = 1
  th.overwrite = True
  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  th.evaluate_test_set = True
  th.gather_classification_results = True

  th.mark = '{}({})_{}'.format(model_name, th.archi_string, th.activation)
  th.gather_summ_name = th.prefix + summ_name + '.sum'
  core.activate()


if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()

