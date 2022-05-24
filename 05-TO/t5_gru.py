from tframe import tf
from tframe import console
from tframe.utils.misc import date_string
from tframe.nets.rnn_cells.gru import GRU
from tframe.utils.organizer.task_tools import update_job_dir

import to_core as core
import to_mu as m


# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'gru'
id = 2
def model(th):
  assert isinstance(th, m.Config)
  cell = GRU(
    state_size=th.state_size,
    use_reset_gate=th.use_reset_gate,
  )
  return m.typical(th, cell)


def main(_):
  console.start('{} on TO task'.format(model_name.upper()))

  th = core.th
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.sequence_length = 100
  th.bits = 3

  # ---------------------------------------------------------------------------
  # 1. folder/file names and device
  # ---------------------------------------------------------------------------
  update_job_dir(id, model_name)
  summ_name = model_name
  th.prefix = '{}_'.format(date_string())
  th.suffix = ''

  th.visible_gpu_id = 0
  # ---------------------------------------------------------------------------
  # 2. model setup
  # ---------------------------------------------------------------------------
  th.model = model
  th.state_size = 100
  th.use_reset_gate = True

  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.max_iterations = 50000

  th.optimizer = tf.train.AdamOptimizer
  th.learning_rate = 0.001
  # ---------------------------------------------------------------------------
  # 4. summary and note setup
  # ---------------------------------------------------------------------------
  th.export_tensors_upon_validation = True

  # th.note_cycle = 300
  # th.validate_cycle = 10
  # th.export_states = True
  # th.export_gates = True
  # th.export_dl_ds_stat = True
  # th.export_dl_dx = True
  # ---------------------------------------------------------------------------
  # 5. other stuff and activate
  # ---------------------------------------------------------------------------
  tail = '_{}bits_L{}'.format(th.bits, th.sequence_length)
  th.mark = '{}({})'.format(model_name, th.state_size) + tail
  th.gather_summ_name = th.prefix + summ_name + tail + th.suffix + '.sum'
  core.activate()


if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()