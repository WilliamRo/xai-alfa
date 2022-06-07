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
from dsc.dsc_config import DSConfig as Hub
from tframe import console
from tframe import Classifier

import dsc_du as du


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
# Configure dataset
# -----------------------------------------------------------------------------
th.val_proportion = 0.2
th.test_proportion = 0.2

# -----------------------------------------------------------------------------
# Set common trainer configs
# -----------------------------------------------------------------------------
th.batch_size = 64
th.num_steps = -1

th.validation_per_round = 2

th.save_model = True
th.gather_note = True



def activate():
  # Load data
  train_set, val_set, test_set = du.load()

  assert callable(th.model)
  model = th.model()
  assert isinstance(model, Classifier)

  # Train
  if th.train:
    model.train(train_set, validation_set=val_set, trainer_hub=th)
  else:
    model.evaluate_model(test_set)

  # End
  model.shutdown()
  console.end()
