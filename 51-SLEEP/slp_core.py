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
from slp.slp_config import SLPConfig as Hub
from tframe import console
from tframe import Classifier

# import slp_du as du


# -----------------------------------------------------------------------------
# Initialize config and set data/job dir
# -----------------------------------------------------------------------------
th = Hub(as_global=True)
th.config_dir()

# -----------------------------------------------------------------------------
# Device configuration
# -----------------------------------------------------------------------------
th.allow_growth = True
th.gpu_memory_fraction = 0.5

# -----------------------------------------------------------------------------
# Set information about the data set
# -----------------------------------------------------------------------------
th.sequence_length = 3840 # 128Hz * 30
th.bits = 3  # 8 classifications

th.input_shape = [6]
th.fixed_length = True
th.val_size = 500   #TODO
th.test_size = 10000   #TODO


# -----------------------------------------------------------------------------
# Set common trainer configs
# -----------------------------------------------------------------------------
th.early_stop = True
th.patience = 10

th.print_cycle = 5
th.validation_per_round = 2

th.export_tensors_upon_validation = True

th.evaluate_train_set = True
th.evaluate_val_set = True
th.evaluate_test_set = True

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

  # Load data
  train_set, val_set, test_set = du.load_data(
    th.data_dir, th.sequence_length, th.bits, th.fixed_length, th.val_size,
    th.test_size)


  # Build model
  assert callable(th.model)
  model = th.model()
  assert isinstance(model, Classifier)

  # Rehearse if required
  if th.rehearse:
    model.rehearse(export_graph=True, build_model=False,
                   path=model.agent.ckpt_dir, mark='model')
    return

  # Train or evaluate
  if th.train:
    model.train(train_set, validation_set=val_set, test_set=test_set,
                trainer_hub=th)
  else:model.evaluate_model(test_set)

  #End
  model.shutdown()
  console.end()
