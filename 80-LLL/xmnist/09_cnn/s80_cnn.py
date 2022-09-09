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
s.register('patience', 5)

s.register('trial_id', 1)
s.register('developer_code', 'o-lr0.0002')

s.register('cl_reg_config', 'l2')
s.register('cl_reg_lambda', 0.0)
# s.register('cl_reg_lambda', 0, 1, 10, 100)

s.register('balance_classes', s.false)

s.register('data_config', 'beta:0.8')
# s.register('data_config', 'alpha:12,2,2,2,2')

s.register('lr', 0.0002)
s.constrain({'train_id': 1}, {'lr': 0.001})
s.register('batch_size', 128)

s.register('train_id', *range(5))
s.run(rehearsal=False)