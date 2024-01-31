import mg_core as core
import mg_mu as m

from tframe import console
from tframe import tf
from tframe.utils.misc import date_string
from tframe.utils.organizer.task_tools import update_job_dir


# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'model17'
id = 2
def model():
  th = core.th
  gan = m.get_container(flatten_D_input=True)

  {'vanilla': m.vanilla17,
   'dcgan': m.dcgan17,
  }[th.archi_string](gan)

  return m.finalize(gan)



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

  th.visible_gpu_id = 0
  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model

  th.archi_string = 'vanilla'

  th.z_dim = 100
  th.use_batchnorm = False
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 1000
  th.batch_size = 128

  th.learning_rate = 0.0001

  th.train = 1
  th.overwrite = True
  th.print_cycle = 20
  th.probe_per_round = 0.1
  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = '{}(z{})'.format(th.archi_string, th.z_dim)
  th.gather_summ_name = th.prefix + summ_name + '.sum'
  core.activate()



if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()

