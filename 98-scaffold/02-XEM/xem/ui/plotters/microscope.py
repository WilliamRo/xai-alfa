import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.axes_grid1 import make_axes_locatable
from pictor.objects.image.large_image import LargeImage
from pictor.plotters.plotter_base import Plotter



class Microscope(Plotter):

  def __init__(self, pictor=None):
    # Call parent's constructor
    super(Microscope, self).__init__(self.show_sample, pictor)

    # Buffers
    self._current_li: LargeImage = None

    # Settable attributes
    self.new_settable_attr('max_size', 1000, int, 'Maximum edge size to plot')

    self.new_settable_attr('color_bar', False, bool, 'Color bar')
    self.new_settable_attr('k_space', False, bool, 'Whether to show k-space')
    self.new_settable_attr('log', False, bool, 'Use log-scale in k-space')
    self.new_settable_attr('vmin', None, float, 'Min value')
    self.new_settable_attr('vmax', None, float, 'Max value')
    self.new_settable_attr('cmap', None, float, 'Color map')
    self.new_settable_attr('interpolation', None, str, 'Interpolation method')
    self.new_settable_attr('title', False, bool, 'Whether to show title')
    self.new_settable_attr('mini_map', False, bool, 'Whether to show mini-map')
    self.new_settable_attr('mini_map_size', 0.3, float, 'Size of mini-map')
    self.new_settable_attr('move_step', 0.2, float, 'Moving step')

  # region: Plot Methods

  def show_sample(self, ax: plt.Axes, x: np.ndarray, fig: plt.Figure, label):
    # Clear axes before drawing, and hide axes
    ax.set_axis_off()

    # If x is not provided
    if x is None:
      self.show_text('No image found', ax)
      return
    li: LargeImage = LargeImage.wrap(x)
    li.thumbnail_size = self.get('max_size')
    x = li.roi_thumbnail

    # Set x to buffer to zoom-in and zoom-out
    self._current_li = li

    # Do 2D DFT if required
    if self.get('k_space'):
      x: np.ndarray = np.abs(np.fft.fftshift(np.fft.fft2(x)))
      if self.get('log'): x: np.ndarray = np.log(x + 1e-10)

    # Show image
    im = ax.imshow(x, cmap=self.get('cmap'),
                   interpolation=self.get('interpolation'),
                   vmin=self.get('vmin'), vmax=self.get('vmax'))

    # Show color bar if required
    if self.get('color_bar'):
      divider = make_axes_locatable(ax)
      cax = divider.append_axes('right', size='5%', pad=0.05)
      fig.colorbar(im, cax=cax)

    # Show title if provided
    if label is not None and self.get('title'): ax.set_title(label)

    # Show mini-map if required
    if self.get('mini_map'): self._show_mini_map(li, ax)


  def _show_mini_map(self, li: LargeImage, ax: plt.Axes):
    # Configs
    r = self.get('mini_map_size')      # ratio
    m = 5                 # margin
    color = 'r'

    H, W = li.roi_thumbnail.shape[:2]
    # Outline
    h, w = int(r * H), int(r * W)
    rect = patches.Rectangle((m, m), w, h, edgecolor=color, facecolor='none',
                             linestyle='-', linewidth=2, alpha=0.5)
    ax.add_patch(rect)
    # ROI
    hr, wr = li.roi_range
    # anchors
    ah, aw = int(m + hr[0] * h), int(m + wr[0] * w)
    # lengths
    lh, lw = [int(s * l) for s, l in zip(li.roi_size, [h, w])]
    rect = patches.Rectangle((aw, ah), lw, lh, edgecolor='none',
                             facecolor=color, linewidth=2, alpha=0.3)
    ax.add_patch(rect)

  # endregion: Plot Methods

  # region: ROI

  def _set_range(self, r, u1, u2):
    assert 0 <= u1 < u2 <= 1

    s = u2 - u1
    d = s * (r - 1.0) / 2
    v1, v2 = u1 - d, u2 + d

    # Apply constrains
    if v1 < 0: v1, v2 = 0, v2 - v1
    if v2 > 1: v1, v2 = v1 - v2 + 1, 1

    return v1, v2

  def zoom(self, ratio: float):
    """Zoom-in when ratio < 1, or zoom-out when ratio > 1
    """
    assert isinstance(ratio, (int, float)) and ratio > 0
    if ratio == 1.0: return

    li: LargeImage = self._current_li

    li.set_roi(*[self._set_range(ratio, *li.roi_range[i]) for i in (0, 1)])
    self.refresh()

  def move_roi(self, h_step: float, w_step: float):
    self._current_li.move_roi(h_step, w_step)
    self.refresh()

  # endregion: ROI

  # region: Private Methods


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
    self.register_a_shortcut(
      'M', lambda: self.flip('mini_map'), 'Turn on/off mini-map')

    # Zoom in/out
    self.register_a_shortcut('O', lambda: self.zoom(2.0), 'Zoom out')
    self.register_a_shortcut('I', lambda: self.zoom(0.5), 'Zoom in')

    # Move ROI
    self.register_a_shortcut(
      'K', lambda: self.move_roi(-self.get('move_step'), 0), 'Move to north')
    self.register_a_shortcut(
      'J', lambda: self.move_roi(self.get('move_step'), 0), 'Move to south')
    self.register_a_shortcut(
      'H', lambda: self.move_roi(0, -self.get('move_step')), 'Move to west')
    self.register_a_shortcut(
      'L', lambda: self.move_roi(0, self.get('move_step')), 'Move to east')

  def set_value(self, vmin: str = None, vmax: str = None):
    """Set minimum value and maximum value"""
    for key, value in zip(['vmin', 'vmax'], [vmin, vmax]):
      try: self.set(key, float(value))
      except: self.set(key, None)
  sv = set_value

  # endregion: Commands
