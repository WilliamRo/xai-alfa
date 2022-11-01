from pictor import Pictor
from topaz.utils.data.loader import load_image
from xem.ui.omma import Omma

import os
import numpy as np



# -----------------------------------------------------------------------------
# Read data
# -----------------------------------------------------------------------------
folder_path = r'../data/EMPIAR-10025/rawdata/micrographs/'
file_names = ['14sep05c_c_00003gr_00014sq_00004hl_00004es_c.mrc']
file_names.append('14sep05c_c_00003gr_00014sq_00005hl_00003es_c.mrc')
file_names.append('14sep05c_c_00003gr_00014sq_00007hl_00004es_c.mrc')

objects, labels = [], []
for fn in file_names:
  file_path = os.path.join(folder_path, fn)
  assert os.path.exists(file_path)
  x = np.array(load_image(file_path), copy=False).astype(np.float32)
  objects.append(x)
  labels.append(fn)


# -----------------------------------------------------------------------------
# Show data in Omma
# -----------------------------------------------------------------------------
from xem.ui.plotters.microscope import Microscope

om = Omma('Omma', figure_size=(8, 8))
om.objects = objects
om.labels = labels

ms = om.add_plotter(Microscope())
ms.set('title', True)
ms.set('color_bar', True)
# ms.set('mini_map', True)
# ms.zoom(0.5)
ms.sv(10, 20)

om.show()


