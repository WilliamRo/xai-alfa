import sys
sys.path.append('../')

from tframe.utils.script_helper import Helper
s = Helper()

from lll_core import th
s.register_flags(type(th))
# -----------------------------------------------------------------------------
# Configure data set here
# -----------------------------------------------------------------------------
pass

# -----------------------------------------------------------------------------
# Specify summary file name and GPU ID here
# -----------------------------------------------------------------------------
summ_name = s.default_summ_name
gpu_id = 0

s.register('gather_summ_name', summ_name + '.sum')
s.register('gpu_id', gpu_id)
s.register('allow_growth', False)
# -----------------------------------------------------------------------------
# Set up your models and run
# -----------------------------------------------------------------------------
s.register('train', True)
s.register('epoch', 1000)
s.register('patience', 5)

s.register('data_config', '2,1,1,1')
# s.register('trial_id', *range(1, 3))
s.register('train_id', *range(4))

s.run(rehearsal=False)