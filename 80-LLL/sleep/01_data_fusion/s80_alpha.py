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
gpu_id = 0

s.register('gather_summ_name', summ_name + '.sum')
s.register('gpu_id', gpu_id)
s.register('allow_growth', False)
# -----------------------------------------------------------------------------
# Set up your models and run
# -----------------------------------------------------------------------------
s.register('train', True)
s.register('epoch', 1000)
s.register('patience', 20)

s.register('cl_reg_config', 'si')
s.register('cl_reg_lambda', 0., 0.5, 1.0)

s.register('trial_id', 0)

s.register('data_config', '35,1,1,1,1')
s.register('train_id', *range(5))

s.run(rehearsal=False)
