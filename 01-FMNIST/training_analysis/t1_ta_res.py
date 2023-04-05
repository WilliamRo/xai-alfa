import fm_core as core
import fm_mu as m

from tframe import console
from tframe import tf


# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'resnet'
id = 2
def model():
  th = core.th
  model = m.get_container(flatten=False)

  for i, c in enumerate(core.th.archi_string.split('-')):
    if c == 'p':
      model.add(m.mu.MaxPool2D(pool_size=2, strides=2))
      continue

    if c[0] == 'r':
      c = int(c[1:])
      vertices = [m.mu.HyperConv2D(c, th.kernel_size, activation=th.activation),
                  m.mu.Merge.Sum()]
      edges = '1;11'
      model.add(m.mu.ForkMergeDAG(vertices, edges, name=f'Residual{i+1}'))

    else:
      c = int(c)
      model.add(m.mu.Conv2D(
        filters=c, kernel_size=th.kernel_size, use_bias=False,
        activation=th.activation))

  # Add flatten layer
  model.add(m.mu.Flatten())
  return m.finalize(model)


def main(_):
  console.start('{} on FMNIST'.format(model_name.upper()))

  th = core.th
  th.rehearse = False
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  pass

  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  th.update_job_dir(id, model_name)
  summ_name = model_name
  th.set_date_as_prefix()

  th.visible_gpu_id = 1
  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model

  th.archi_string = '32-p-r32-r32-r32-r32'
  th.kernel_size = 3
  th.activation = 'relu'
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 1
  th.batch_size = 128

  th.optimizer = 'adam'
  th.learning_rate = 0.0003

  th.train = True
  th.overwrite = True
  th.print_cycle = 20
  # ---------------------------------------------------------------------------
  # 4. monitor setup
  # ---------------------------------------------------------------------------
  th.validate_at_the_beginning = True
  th.validation_per_round = 100
  th.export_weights = True

  th.suffix = '_001'
  # ---------------------------------------------------------------------------
  # 5. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = '{}({})'.format(model_name, th.archi_string)
  th.gather_summ_name = th.prefix + summ_name + '.sum'
  core.activate()


if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()

