from pictor import Pictor
from topaz.utils.data.loader import load_image
from xem.ui.omma import Omma

import os
import numpy as np



# -----------------------------------------------------------------------------
# Read data
# -----------------------------------------------------------------------------
file_path = r'../data/EMPIAR-10025/rawdata/micrographs/'
file_path += r'14sep05c_c_00003gr_00014sq_00004hl_00004es_c.mrc'

assert os.path.exists(file_path)
x = np.array(load_image(file_path), copy=False).astype(np.float32)


# -----------------------------------------------------------------------------
# Show data in Omma
# -----------------------------------------------------------------------------
from pictor.plotters.retina import Retina
from xem.ui.plotters.microscope import Microscope

om = Omma('Omma', figure_size=(7, 7))
om.objects = [x]
# om.add_plotter(Retina())
om.add_plotter(Microscope())
om.show()


