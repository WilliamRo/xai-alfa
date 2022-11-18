from pictor.plotters import Monitor
from pictor.objects.signals.scrolling import Scrolling

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List
from pictor.plotters.plotter_base import Plotter
from pictor.objects.signals import SignalGroup

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np



class MonitorPro(Monitor):

  def __init__(self, pictor=None, window_duration=60, channels: str='*'):
    """
    :param window_duration: uses second as unit
    """
    # Call parent's constructor
    super(Monitor, self).__init__(self.show_curves, pictor)

    # Specific attributes
    self.scroll_buffer = {}
    self._selected_signal: Optional[Scrolling] = None

    # Settable attributes
    self.new_settable_attr('default_win_duration', window_duration,
                           float, 'Default window duration')
    self.new_settable_attr('step', 0.1, float, 'Window moving step')
    self.new_settable_attr('bar', True, bool,
                           'Whether to show a location bar at the bottom')
    self.new_settable_attr('channels', channels, str,
                           'Channels to display, `all` by default')
    self.new_settable_attr('max_ticks', 10000, int, 'Maximum ticks to plot')
    self.new_settable_attr('smart_scale', True, bool,
                           'Whether to use smart scale ')
    self.new_settable_attr('xi', 0.1, float, 'Margin for smart scale')
    self.new_settable_attr('hl', 0, int, 'Highlighted channel id')
    self.new_settable_attr('annotation', None, str, 'Annotation key')

    self.new_settable_attr('psg', False, bool, 'Whether to show PSG')

  def show_curves(self, x: np.ndarray, fig: plt.Figure, i: int):
    # Clear figure
    fig.clear()

    # If x is not provided
    if x is None:
      self.show_text('No signal found', fig=fig)
      return

    # Get a Scrolling object based on input x
    s = self._get_scroll(x, i)
    self._selected_signal = s

    # Create subplots
    height_ratios = [50]
    if self.get('bar'): height_ratios.append(1)
    axs = fig.subplots(
      len(height_ratios), 1, gridspec_kw={'height_ratios': height_ratios})

    # Plot signals
    ax: plt.Axes = axs[0] if len(height_ratios) > 1 else axs
    self._plot_curve(ax, s)

    # Plot annotations
    self._plot_annotation(ax, s)

    # Show scroll-bar if necessary
    if self.get('bar'): self._outline_bar(axs[-1], s)

  def _plot_curve(self, ax: plt.Axes, s: Scrolling):
    """ i  y
           2  ---------
        0     -> N(=2) - i(=0) - 0.5 = 1.5
           1  ---------
        1     -> N(=2) - i(=1) - 0.5 = 0.5
           0  ---------
    """
    # Get settings
    smart_scale = self.get('smart_scale')
    hl_id = self.get('hl')

    # Get channels [(name, x, y)]
    channels = s.get_channels(self.get('channels'),
                              max_ticks=self.get('max_ticks'))
    N = len(channels)

    margin = 0.05
    for i, (name, x, y) in enumerate(channels):
      # Normalized y before plot
      if not smart_scale:
        y = y - min(y)
        y = y / max(y) * (1.0 - 2 * margin) + margin
      else:
        xi = self.get('xi')
        mi = s.get_channel_percentile(name, xi)
        ma = s.get_channel_percentile(name, 100 - xi)
        y = y - mi
        y = y / (ma - mi) * (1.0 - 2 * margin) + margin

      y = y + N - 1 - i
      # Plot normalized y
      color, zorder = 'black', 10
      if 0 < hl_id != i + 1: color, zorder = '#AAA', None
      ax.plot(x, y, color=color, linewidth=1, zorder=zorder)

      # Set xlim
      if i == 0: ax.set_xlim(x[0], x[-1])

    # Set y_ticks
    ax.set_yticks([N - i - 0.5 for i in range(N)])
    ax.set_yticklabels([name for name, _, _ in channels])

    # Highlight label if necessary
    if hl_id > 0:
      for i, label in enumerate(ax.get_yticklabels()):
        label.set_color('black' if i + 1 == hl_id else 'grey')

    # Set styles
    ax.set_ylim(0, N)
    ax.grid(color='#E03', alpha=0.4)

    tail = f' (xi={self.get("xi")})' if smart_scale else ''
    ax.set_title(s.label + tail)

