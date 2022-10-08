import cv2
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.axes_grid1 import make_axes_locatable
from pictor.plotters.plotter_base import Plotter




class Microscope(Plotter):

  def __init__(self, pictor=None):
    # Call parent's constructor
    super(Microscope, self).__init__(self.show_sample, pictor)

    # Settable attributes
    self.new_settable_attr('max_size', 1000, int, 'Maximum edge size to plot')

    self.new_settable_attr('color_bar', False, bool, 'Color bar')
    self.new_settable_attr('k_space', False, bool, 'Whether to show k-space')
    self.new_settable_attr('log', False, bool, 'Use log-scale in k-space')
    self.new_settable_attr('alpha', None, float, 'Alpha value')
    self.new_settable_attr('vmin', None, float, 'Min value')
    self.new_settable_attr('vmax', None, float, 'Max value')
    self.new_settable_attr('cmap', None, float, 'Color map')
    self.new_settable_attr('interpolation', None, str, 'Interpolation method')
    self.new_settable_attr('title', False, bool, 'Whether to show title')


  def show_sample(self, ax: plt.Axes, x: np.ndarray, fig: plt.Figure):
    # Clear axes before drawing, and hide axes
    ax.set_axis_off()

    # If x is not provided
    if x is None:
      self.show_text('No image found', ax)
      return
    x = self._shrink_img(x)

    # Do 2D DFT if required
    if self.get('k_space'):
      x: np.ndarray = np.abs(np.fft.fftshift(np.fft.fft2(x)))
      if self.get('log'): x: np.ndarray = np.log(x + 1e-10)

    # Show image
    im = ax.imshow(x, cmap=self.get('cmap'), alpha=self.get('alpha'),
                   interpolation=self.get('interpolation'),
                   vmin=self.get('vmin'), vmax=self.get('vmax'))

    # Show color bar if required
    if self.get('color_bar'):
      divider = make_axes_locatable(ax)
      cax = divider.append_axes('right', size='5%', pad=0.05)
      fig.colorbar(im, cax=cax)

  # region: Private Methods

  def _shrink_img(self, im: np.ndarray):
    """Shrink image according to `max_size`.
    im: should be of shape [H, W, ...]
    """
    max_size = self.get('max_size')
    if max_size is None: return im

    H, W = im.shape[:2]
    if max(H, W) <= max_size: return im
    if H > W:
      h = max_size
      w = int(W / H * h)
    else:
      w = max_size
      h = int(H / W * w)

    # To shrink an image, it will generally look best with #INTER_AREA
    im = cv2.resize(im, dsize=(h, w), interpolation=cv2.INTER_AREA)

    return im

  # endregion: Private Methods

  # region: Commands

  def register_shortcuts(self):
    self.register_a_shortcut(
      'T', lambda: self.flip('title'), 'Turn on/off title')
    self.register_a_shortcut(
      'C', lambda: self.flip('color_bar'), 'Turn on/off color bar')
    self.register_a_shortcut(
      'F', lambda: self.flip('k_space'), 'Turn on/off k-space view')
    self.register_a_shortcut(
      'L', lambda: self.flip('log'),
      'Turn on/off log scale in k-space view')

  # endregion: Commands
