import md_core as core
import md_mu as m

from tframe import console
from tframe import tf
from tframe.utils.misc import date_string
from tframe.utils.organizer.task_tools import update_job_dir


# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'unet'
id = 1
def model():
  th = core.th
  model = m.get_container(time_steps=100)

  unet = m.TimeUNet2D(th.archi_string, model)
  unet.add_to(model)

  return m.finalize(model)



def main(_):
  console.start('{} on FMNIST task'.format(model_name.upper()))

  th = core.th
  th.rehearse = 0
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
  th.suffix = '_2'

  th.visible_gpu_id = 0
  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model

  th.archi_string = '8-3-2-2-relu-mp'
  th.beta_schedule = 'cosine'
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 100
  th.batch_size = 128

  th.optimizer = 'adam'
  th.learning_rate = 0.0001
  # th.lr_decay_method = 'cos'

  th.train = 0
  th.overwrite = 1
  th.print_cycle = 20
  th.probe_per_round = 1

  th.patience = 10
  th.early_stop = False
  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = '{}({},{})'.format(model_name, th.archi_string, th.beta_schedule)
  th.gather_summ_name = th.prefix + summ_name + '.sum'
  core.activate()



if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()

