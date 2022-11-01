from pictor import Pictor
from topaz.utils.data.loader import load_image
from xem.ui.omma import Omma

import os
import numpy as np
import mrcfile



# -----------------------------------------------------------------------------
# Read data
# -----------------------------------------------------------------------------
folder_path = r'../data/Tomo110/frames'
file_names = ['01_Tomo110_64.0_Apr04_15.51.42.mrc']
# file_names.append('')

objects, labels = [], []
for fn in file_names:
  file_path = os.path.join(folder_path, fn)
  assert os.path.exists(file_path)
  with mrcfile.open(fn) as mrc:
    # TODO: currently movie files can not be read
    pass


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

om.show()


