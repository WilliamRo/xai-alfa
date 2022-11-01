from tframe import tf
from tframe import console
from tframe.utils.misc import date_string
from tframe.utils.organizer.task_tools import update_job_dir

import slp_core as core
import slp_mu as m


# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'il_data_fusion'
id = 3
def model(): return m.get_data_fusion_model()


def main(_):
  console.start('{} on Sleep task'.format(model_name.upper()))

  th = core.th
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.data_config = ['7', '3,2,2'][1]
  th.aug_config = '2*3'
  th.channels = '0,1,2'

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

  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 10000
  th.train = True

  th.patience = 2
  th.optimizer = tf.train.AdamOptimizer
  th.learning_rate = 0.003
  # ---------------------------------------------------------------------------
  # 4. incremental learning arguments
  # ---------------------------------------------------------------------------
  th.train_id = 0
  th.overwrite = th.train_id == 0

  th.cl_reg_config = 'si'
  th.cl_reg_lambda = 1.0

  assert ',' in th.data_config
  # ---------------------------------------------------------------------------
  # 5. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = f'{model_name}(c-{th.channels})'
  th.gather_summ_name = th.prefix + summ_name + '-df.sum'
  core.activate()


if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()
