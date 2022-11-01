from tframe import tf
from tframe import console
from tframe.utils.misc import date_string
from tframe.utils.organizer.task_tools import update_job_dir

import slp_core as core
import slp_mu as m


# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'feature_fusion'
id = 2
def model(): return m.get_feature_fusion_model()


def main(_):
  console.start('{} on Sleep task'.format(model_name.upper()))

  th = core.th
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.data_config = ['7', '3,2,2'][0]
  th.aug_config = '2*3'
  th.channels = '0,1;2'

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
  th.train = True

  th.epoch = 1
  th.optimizer = tf.train.AdamOptimizer
  th.learning_rate = 0.003
  # ---------------------------------------------------------------------------
  # 4. summary and note setup
  # ---------------------------------------------------------------------------

  # ---------------------------------------------------------------------------
  # 5. other stuff and activate
  # ---------------------------------------------------------------------------
  c_str = f'{th.channels}'.replace(';', '-')
  th.mark = f'{model_name}(c-{c_str})'
  th.gather_summ_name = th.prefix + summ_name + '-ff.sum'
  core.activate()


if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()
