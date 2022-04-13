import os
import cv2

from pictor import DaVinci
from roster import get_xai_alfa_dir
from tframe.data.images.cifar10 import CIFAR10



# -----------------------------------------------------------------------------
#  Load data
# -----------------------------------------------------------------------------
DATA_PATH = os.path.join(get_xai_alfa_dir(), 'data', 'cifar10')
# Borrow tframe.CIFAR10 class to load data
train_set, _, _ = CIFAR10.load(DATA_PATH, one_hot=False)
# Get images to show
images = train_set.features / 255.0


# -----------------------------------------------------------------------------
#  Initialize a DaVinci
# -----------------------------------------------------------------------------
da = DaVinci('CIFAR-10 Viewer', init_as_image_viewer=True)


# -----------------------------------------------------------------------------
#  Define a plotter and add to `da`
# -----------------------------------------------------------------------------
def plotter(x, ax):
  # Here x is the current object
  # Do something to x
  y = cv2.rotate(x, cv2.cv2.ROTATE_180)
  # Plot
  ax.imshow(y)
  ax.axis('off')

da.add_plotter(plotter)


# -----------------------------------------------------------------------------
#  Set images and display
# -----------------------------------------------------------------------------
# Set images
da.objects = images
# Set image titles
da.object_titles = [train_set['CLASSES'][c[0]] for c in train_set.targets]
# Show GUI
da.show()


