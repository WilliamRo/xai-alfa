import fm_core as core
import fm_mu as m

from tframe import console
from tframe import tf
from tframe.utils.misc import date_string
from tframe.utils.organizer.task_tools import update_job_dir


# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'nas101'
id = 4
def model():
  th = core.th
  model = m.get_container(flatten=False)
  m.mu.NAS101(vertices=th.vertices.split(','), edges=th.adj_matrix,
              num_stacks=th.num_stacks, stem_channels=th.filters,
              cells_per_stack=th.module_per_stack,
              use_batchnorm=th.use_batchnorm,
              input_projection=th.input_projection).add_to(model)

  # Add flatten layer
  return m.finalize(model)


def main(_):
  console.start('{} on CIFAR-10 task'.format(model_name.upper()))

  th = core.th
  th.rehearse = False
  # ---------------------------------------------------------------------------
  # 0. date set setup
  # ---------------------------------------------------------------------------
  th.centralize_data = False

  th.augmentation = True
  th.aug_config = 'flip:True;False'
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

  th.vertices, th.adj_matrix = m.mu.NAS101.Shelf.CIFAR10.NAS101Best
  th.filters = 16
  th.num_stacks = 3
  th.module_per_stack = 2
  th.input_projection = False
  th.use_batchnorm = True
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 1000
  th.batch_size = 32

  th.optimizer = 'adam'
  th.learning_rate = 0.003
  th.patience = 15

  th.train = True
  th.overwrite = True
  th.print_cycle = 20
  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = '{}(f{}-ns{}-mps{})'.format(
    model_name, th.filters, th.num_stacks, th.module_per_stack)
  th.gather_summ_name = th.prefix + summ_name + '.sum'
  core.activate()


if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()

