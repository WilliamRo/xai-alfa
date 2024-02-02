import mg_core as core
import mg_mu as m

from tframe import console
from tframe import tf
from tframe.utils.misc import date_string
from tframe.utils.organizer.task_tools import update_job_dir


# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'simple'
id = 1
def model():
  th = core.th
  gan = m.get_container(flatten_D_input=True)

  # Build generator
  H, N = 128, 4
  for _ in range(N):
    gan.G.add(m.mu.HyperDense(H))
    if th.use_batchnorm: gan.G.add(m.mu.BatchNorm())
    gan.G.add(m.mu.Activation('relu'))
    H = H * 2
  gan.G.add(m.mu.HyperDense(784, activation='sigmoid'))
  gan.G.add(m.mu.Reshape(shape=th.input_shape))

  # Build discriminator
  H, N = 128, 3
  H = H * (2 ** N)
  for _ in range(N):
    H = H // 2
    gan.D.add(m.mu.HyperDense(H, activation='lrelu'))

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
  th.suffix = '_long'

  th.visible_gpu_id = 0
  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model

  th.z_dim = 64
  th.use_batchnorm = True
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 1000
  th.batch_size = 100

  th.learning_rate = 0.00001

  th.train = 1
  th.overwrite = True
  th.print_cycle = 20
  th.probe_per_round = 0.2

  th.epoch = 1000
  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = '{}(z{})'.format(model_name, th.z_dim)
  th.gather_summ_name = th.prefix + summ_name + '.sum'
  core.activate()



if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()

