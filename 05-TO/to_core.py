import sys, os
#: Add necessary paths to system path list so that all task modules with
#:  filename `tXX_YYY.py` can be run directly.
#:
#: Recommended project structure:
#: DEPTH  0          1         2 (*)
#:        this_proj
#:                |- 01-MNIST
#:                          |- mn_core.py
#:                          |- mn_du.py
#:                          |- mn_mu.py
#:                          |- t1_lenet.py
#:                |- 02-CIFAR10
#:                |- ...
#:                |- tframe
#:
#! Specify the directory depth with respect to the root of your project here
DIR_DEPTH = 2
ROOT = os.path.abspath(__file__)
for _ in range(DIR_DEPTH):
  ROOT = os.path.dirname(ROOT)
  if sys.path[0] != ROOT: sys.path.insert(0, ROOT)
# =============================================================================
from tframe import console
from tframe import DefaultHub as Hub
from tframe import Classifier

import to_du as du


# -----------------------------------------------------------------------------
# Initialize config and set data/job dir
# -----------------------------------------------------------------------------
th = Hub(as_global=True)
th.config_dir()

# -----------------------------------------------------------------------------
# Device configuration
# -----------------------------------------------------------------------------
th.allow_growth = False
th.gpu_memory_fraction = 0.30

# -----------------------------------------------------------------------------
# Set information about the data set
# -----------------------------------------------------------------------------
th.sequence_length = 100
th.bits = 3

th.input_shape = [6]
th.fixed_length = True
th.val_size = 500
th.test_size = 10000

# -----------------------------------------------------------------------------
# Set common trainer configs
# -----------------------------------------------------------------------------
th.batch_size = 20

th.validate_cycle = 20

th.num_steps = -1
th.val_batch_size = -1
th.print_cycle = 1
th.sample_num = 2

th.save_model = False
th.gather_note = True


def activate():
  # This line must be put in activate
  th.output_dim = 2 ** th.bits

  assert callable(th.model)
  model = th.model(th)
  assert isinstance(model, Classifier)

  # Load data
  train_set, val_set, test_set = du.load(
    th.data_dir, th.sequence_length, th.bits, th.fixed_length, th.val_size,
    th.test_size)

  # Train
  if th.train:
    model.train(train_set, validation_set=val_set, trainer_hub=th,
                terminator=lambda metric: metric == 1.0)
  else: model.evaluate_model(test_set)

  # End
  model.shutdown()
  console.end()
