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
gpu_id = 0

s.register('gather_summ_name', summ_name + '.sum')
s.register('gpu_id', gpu_id)
s.register('allow_growth', False)
# -----------------------------------------------------------------------------
# Set up your models and run
# -----------------------------------------------------------------------------
s.register('train', True)
s.register('epoch', 10)

s.register('lr', 0.0001, 0.001, 0.01)
s.register('batch_size', 32, 128, 512)

# s.configure_engine(times=10)
s.configure_engine(strategy='skopt', criterion='Best Accuracy')
s.run(rehearsal=False)