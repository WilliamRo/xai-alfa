import sys
sys.path.append('../../')

from tframe.utils.script_helper import Helper
s = Helper()

from lll_core import th
s.register_flags(type(th))
# -----------------------------------------------------------------------------
# Configure data set here
# -----------------------------------------------------------------------------
pass

# ----------------------------------------------------------------------------
# Specify summary file name and GPU ID here
# -----------------------------------------------------------------------------
summ_name = s.default_summ_name
gpu_id = 1

s.register('gather_summ_name', summ_name + '.sum')
s.register('gpu_id', gpu_id)
s.register('allow_growth', False)
# -----------------------------------------------------------------------------
# Set up your models and run
# -----------------------------------------------------------------------------
s.register('train', True)
s.register('epoch', 1000)
s.register('patience', 5)

s.register('trial_id', 5)

s.register('cl_reg_lambda', 0.0)
s.register('cl_reg_config', 'l2')

s.register('data_config', 'beta:0.8')

s.register('lr', 0.001)
s.register('batch_size', 128)

prune = True
if prune:
  # Prune
  s.register('pruning_iterations', *range(10))
  s.register('pruning_rate', 0.2)
  s.register('train_id', 0)
  s.register('developer_code', 'prune')
else:
  # Pull
  s.register('pruning_rate', 0.0)
  s.register('train_id', *range(0, 5))
  s.constrain({'train_id': 0},
              {'mark_to_load': '0908_mark(64-p-32)_pr10'})
  s.register('developer_code', 'pull')

s.run(rehearsal=False)