from tframe import Predictor, pedia, context
from tframe.utils.file_tools.imp_tools import import_from_path

from fm_core import th
from fmnist.fm_agent import FMNIST
from pictor.plotters.retina import Retina



# -----------------------------------------------------------------------------
# 1. Load t-file configures
# -----------------------------------------------------------------------------
t_file_path = r'E:\xai-alfa\01-FMNIST\02_cnn\checkpoints\1104_cnn(64-p-128)_gpu_2_2\1104_cnn(64-p-128)_gpu_2_2.py'
mod = import_from_path(t_file_path)

th.developer_code += 'deactivate'

# Execute main to load basic module settings
mod.main(None)

# -----------------------------------------------------------------------------
# 2. Load data
# -----------------------------------------------------------------------------
train_set, val_set, test_set = FMNIST.load(th.data_dir)

# -----------------------------------------------------------------------------
# 3. Build model and find tensor to export
# -----------------------------------------------------------------------------
model: Predictor = th.model()

tensor_list = [layer.output_tensor for layer in model.layers
               if 'conv' in layer.full_name]

# -----------------------------------------------------------------------------
# 4. Run model to get tensors
# -----------------------------------------------------------------------------
values = model.evaluate(tensor_list, train_set[:5], batch_size=1, verbose=True)

# -----------------------------------------------------------------------------
# 5. Visualize tensor in Pictor
# -----------------------------------------------------------------------------
objects, labels = [], []
index = 2
for v, t in zip(values, tensor_list):
  num_channels = v.shape[-1]
  for i in range(num_channels):
    objects.append(v[index, ..., i])
    labels.append(f'{t.name}[{i}]')

Retina.plot(objects, labels, 'Tensor Visualizer')


