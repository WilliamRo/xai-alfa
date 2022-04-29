import sys
sys.path.append('../')
sys.path.append('../../')

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
s.register('epoch', 1000)
s.register('patience', 10)

s.register('lr', 0.0001, 0.001, 0.01)
s.register('batch_size', 32, 128, 512)
s.register('archi_string', '128', '258', '128-64', '256-128', '128-64-32')
s.register('dropout', 0.0, 0.2, 0.4, 0.6)

s.configure_engine(times=2)
s.run(rehearsal=False)