import sys
sys.path.append('../')

from tframe.utils.script_helper import Helper
s = Helper()

from fm_core import th
s.register_flags(type(th))
# -----------------------------------------------------------------------------
# Configure data set here
# -----------------------------------------------------------------------------
pass

# -----------------------------------------------------------------------------
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
s.register('patience', 10)

s.register('lr', 0.0001, 0.001, 0.01)
s.register('batch_size', 32, 128, 512)
# s.register('archi_string', '64-p-128', '32-p-64', '32-32-p-64-64',
#            '32-32-p-64-64-p-128-128')
s.register('archi_string', '64-p-128', '32-p-64')
s.register('suffix', '_gpu_2_2')
s.register('use_batchnorm', True, False)

# s.configure_engine(times=10)
s.configure_engine(strategy='skopt', criterion='Best Accuracy')
s.run(rehearsal=False)