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
from slp_config import SLPConfig as Hub
from tframe import console

import slp_du as du



# -----------------------------------------------------------------------------
# Initialize config and set data/job dir
# -----------------------------------------------------------------------------
th = Hub(as_global=True)
th.config_dir()
th.data_dir = r'E:\xai-alfa\51-SLEEP\data\sleepedf-lll'

# -----------------------------------------------------------------------------
# Device configuration
# -----------------------------------------------------------------------------
th.allow_growth = False
th.gpu_memory_fraction = 0.5

# -----------------------------------------------------------------------------
# Set common trainer configs
# -----------------------------------------------------------------------------
th.epoch = 10000
th.early_stop = True
th.patience = 10

th.print_cycle = 5
th.validation_per_round = 2

th.export_tensors_upon_validation = True

th.batch_size = 32

th.save_model = True
th.gather_note = True
th.epoch_as_step = False   # for plotting curve

th.evaluate_test_set = True
th.eval_batch_size = 200


def activate():
  # (1) Build model
  model = th.model()

  # (2) Load data
  # ALL: data_sets = [[train1, val1, test1]]
  # LLL: data_sets = [[train1, val1, test1], ..., [trainN, valN, testN]]
  data_sets = du.load_data()

  # (3) Train or evaluate
  train_set, val_set, test_set = data_sets[th.train_id]
  if not th.is_all_data_scene:
    # (3.1) For incremental learning scenario
    test_sets = [ds for _, _, ds in data_sets]
    th.additional_datasets_for_validation.extend(test_sets)
    th.clear_records_before_training = True

  if th.train: model.train(
    train_set, validation_set=val_set, test_set=test_set, trainer_hub=th)

  # End
  model.shutdown()
  console.end()
