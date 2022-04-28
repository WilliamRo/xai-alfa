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
gpu_id = 1

s.register('gather_summ_name', summ_name + '.sum')
s.register('gpu_id', gpu_id)
s.register('gpu_memory_fraction', 0.7)
s.register('allow_growth', False)
# -----------------------------------------------------------------------------
# Set up your models and run
# -----------------------------------------------------------------------------
s.register('train', True)
s.register('epoch', 1000)

s.register('augmentation', s.true_and_false)

s.register('filters', 16, 48)
s.register('num_stacks', 2, 4)
s.register('input_projection', s.true_and_false)

s.register('lr', 0.0001, 0.01)
s.register('batch_size', 16, 256)

s.configure_engine(strategy='skopt', criterion='Best Accuracy')
s.run(rehearsal=False)