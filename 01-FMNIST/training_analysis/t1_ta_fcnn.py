import fm_core as core
import fm_mu as m

from callback_model_updated import put_statistics_to_note
from tframe import console
from tframe import context
from tframe import tf

# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'fcnn'
id = 3
def model():
  th = core.th
  model = m.get_container(flatten=False)

  # Add conv2d and pooling at the top
  model.add(m.mu.Conv2D(
    filters=th.filters, kernel_size=th.kernel_size, use_bias=False,
    activation=th.activation))
  model.add(m.mu.MaxPool2D(pool_size=2, strides=2))

  # Add harp
  vertices = [m.mu.Conv2D(filters=th.filters, kernel_size=th.kernel_size,
                          use_bias=False, activation=th.activation)
              for _ in range(th.num_layers)]
  vertices.append(m.mu.Merge.Sum())

  edges = ''
  for i in range(th.num_layers): edges += '0' * i + '1;'
  edges += '1' * (th.num_layers + 1)

  model.add(m.mu.ForkMergeDAG(vertices, edges, name=f'Harp'))

  return m.finalize(model, fully_conv=True)


def main(_):
  console.start('{} on FMNIST'.format(model_name.upper()))

  th = core.th
  th.rehearse = False
  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  th.update_job_dir(id, model_name)
  summ_name = model_name
  th.set_date_as_prefix()

  th.visible_gpu_id = 0
  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model

  th.filters = 32
  th.num_layers = 5
  th.kernel_size = 5
  th.activation = 'relu'
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 1
  th.batch_size = 128

  th.optimizer = 'adam'
  th.learning_rate = 0.0001

  th.train = True
  th.overwrite = True
  th.print_cycle = 20
  # ---------------------------------------------------------------------------
  # 4. monitor setup
  # ---------------------------------------------------------------------------
  context.depot['callback_model_updated'] = put_statistics_to_note

  th.stats_max_length = 5
  th.monitor_weight_history = True
  th.monitor_weight_grads = True

  th.validate_at_the_beginning = True
  th.validation_per_round = 100

  th.suffix = '_apr6'
  th.developer_code = '04'
  # ---------------------------------------------------------------------------
  # 5. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = '{}({})'.format(model_name, th.archi_string)
  th.gather_summ_name = th.prefix + summ_name + '.sum'
  core.activate()


if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()

