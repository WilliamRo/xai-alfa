import lll_core as core
import lll_mu as m

from tframe.enums import SaveMode
from tframe import console
from tframe import tf
from tframe.utils.misc import date_string
from tframe.utils.organizer.task_tools import update_job_dir


# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'feature_fusion_small'
id = 3
def model():
  from tframe.nets.octopus import Octopus

  th = core.th

  # get_container
  model = m.Classifier(mark=th.mark)
  oc: Octopus = model.add(Octopus())

  def get_extraction_part(name):
    conv = lambda c, k, d=1: m.mu.Conv1D(
      filters=c, kernel_size=k, dilation_rate=d,
      use_batchnorm=th.use_batchnorm, activation=th.activation)
    pool88 = lambda: m.mu.MaxPool1D(pool_size=8, strides=8)
    n = 22
    fm = m.mu.ForkMergeDAG(vertices=[
      [conv(2*n, 50), pool88(), m.mu.Dropout(0.5),
       conv(4*n, 8), conv(4*n, 8), conv(4*n, 8), pool88()],
      [conv(2*n, 580), pool88(), m.mu.Dropout(0.5),
       conv(4*n, 6), conv(4*n, 6), conv(4*n, 6), pool88()],
      [m.mu.Merge.Sum(), m.mu.Dropout(0.5)],
      [conv(n, 6, 5), conv(2*n, 6, 6), conv(4*n, 6, 5), m.mu.Dropout(0.5)],
      [m.mu.Merge.Sum(),m.mu.Dropout(0.5)]],
      edges='1;10;011;0001;00011', name=name)
    return fm

  # EEG channels
  li = oc.init_a_limb('input-1', [3000, 2])
  li.add(get_extraction_part('DAG-1'))

  # EOG channel
  li = oc.init_a_limb('input-2', [3000, 1])
  li.add(get_extraction_part('DAG-2'))

  # Open both gates
  oc.set_gates([1, 1])

  # Add flatten layer
  model.add(m.mu.Flatten())
  return m.finalize(model)


def main(_):
  console.start('{} on LLL task'.format(model_name.upper()))

  th = core.th
  th.rehearse = False
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.task = th.Tasks.SLEEPEDF
  th.developer_code = 'feature'

  th.data_config = '39'
  th.train_id = 0

  th.output_dim = 5
  th.input_shape = [3000, 3]
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

  th.activation = 'relu'
  th.use_batchnorm = True
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 1000
  th.batch_size = 32
  th.balance_classes = False

  th.optimizer = 'adam'
  th.learning_rate = 0.003
  th.patience = 10

  th.validation_per_round = 2
  th.val_progress_bar = True

  th.train = True
  th.overwrite = True if th.train_id == 0 else False
  th.save_mode = SaveMode.ON_RECORD
  th.print_cycle = 10
  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = '{}({})'.format(model_name, th.archi_string)
  th.gather_summ_name = th.prefix + summ_name + '.sum'
  core.activate()


if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()

