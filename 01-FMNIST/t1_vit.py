import fm_core as core
import fm_mu as m

from tframe import console
from tframe import tf
from tframe.utils.misc import date_string
from tframe.utils.organizer.task_tools import update_job_dir

from tframe.layers.vit.partition import Partition
from tframe.layers.vit.patch_encoder import PatchEncoder


# -----------------------------------------------------------------------------
# Define model here
# -----------------------------------------------------------------------------
model_name = 'vit'
id = 5
def model():
  th = core.th
  model = m.get_container(flatten=False)

  model.add(Partition(patch_size=th.int_para_1))
  model.add(PatchEncoder(
    dim=th.hidden_dim, use_positional_embedding=th.bool_para_1))

  for _ in range(th.num_layers):
    x1 = model.add(m.mu.LayerNormalization())
    model.add(m.mu.MultiHeadSelfAttention(th.num_heads, use_keras=1,
                                          dropout=th.dropout))
    x2 = model.add(m.mu.ShortCut(x1, mode='sum'))

    model.add(m.mu.LayerNormalization())
    model.add(m.mu.Dense(th.hidden_dim, activation=th.activation))
    if th.dropout > 0: model.add(m.mu.Dropout(1 - th.dropout))

    model.add(m.mu.ShortCut(x2, mode='sum'))

  # Add flatten layer
  model.add(m.mu.Flatten())
  # model.add(m.mu.Dropout(0.5))
  model.add(m.mu.Dense(128))
  return m.finalize(model)



def main(_):
  console.start('{} on FMNIST'.format(model_name.upper()))

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

  # patch size (28 // 4 = 7)
  th.int_para_1 = 4
  # Use positional embedding
  th.bool_para_1 = True
  th.num_heads = 4
  th.hidden_dim = 32
  th.dropout = 0.0

  th.num_layers = 2
  # ---------------------------------------------------------------------------
  # 3. trainer setup
  # ---------------------------------------------------------------------------
  th.epoch = 100000
  th.batch_size = 128

  th.optimizer = 'adam'
  th.learning_rate = 0.003
  th.patience = 5

  th.activation = 'relu'

  th.train = True
  th.overwrite = True
  th.print_cycle = 200
  th.validation_per_round = 1
  th.validate_train_set = True
  # ---------------------------------------------------------------------------
  # 4. other stuff and activate
  # ---------------------------------------------------------------------------
  th.mark = f'{model_name}(p{th.int_para_1}d{th.hidden_dim})'
  th.gather_summ_name = th.prefix + summ_name + '.sum'
  core.activate()


if __name__ == '__main__':
  console.suppress_logging()
  tf.app.run()

