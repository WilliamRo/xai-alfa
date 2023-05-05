from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pictor.pictor import Plotter
from tframe.utils.note import Note

import matplotlib.pyplot as plt
import numpy as np



class EncoderAnalyzer(Plotter):

  def __init__(self, pictor=None):
    # Call parent's constructor
    super().__init__(self.sort_and_plot)

    # Define settable attributes
    self.new_settable_attr('cmap', 'RdBu', str, 'Color map')
    self.new_settable_attr('wmax', 0.04, float, 'Cut-off value of weights')
    self.new_settable_attr('xmax', 0.14, float,
                           'Max value of x-axis in scatter plot')
    self.new_settable_attr('ymax', 22, float,
                           'Max value of y-axis in scatter plot')

    self.new_settable_attr('xper', 50, float, 'Minimum x-axis percentile')
    self.new_settable_attr('frequp', True, bool,
                           'Whether to sort weights by frequency')
    self.new_settable_attr('filter', True, bool,
                           'Whether to apply percentile filter')

  # region: Properties

  @property
  def selected_w(self):
    return self.pictor.get_element(self.pictor.Keys.OBJECTS)

  @property
  def frequency(self):
    # deprecated
    w = self.selected_w
    der1 = np.sign(w[:, :-1] - w[:, 1:])
    der2 = np.sign(der1[:, :-1] - der1[:, 1:])
    return np.sum(np.abs(der2), axis=-1)

  @property
  def sign_frequency(self):
    return self._calc_freq(self.selected_w)

  @property
  def amplitude(self):
    return self._calc_amp(self.selected_w)

  # endregion: Properties

  # region: Private Methods

  def _calc_freq(self, w):
    w = np.sign(w)
    der1 = np.sign(w[:, :-1] - w[:, 1:])
    return np.sum(np.abs(der1), axis=-1).astype(int)

  def _calc_amp(self, w):
    return np.max(w, axis=-1) - np.min(w, axis=-1)

  # endregion: Private Methods

  # region: Plotting Method

  def sort_rows(self, w: np.ndarray):
    result = []
    freqs = self._calc_freq(w)

    freq_list = (list(range(min(freqs), max(freqs) + 1))
                 if self.get('frequp') else [0])

    for freq in freq_list:
      if freq > 0:
        indices = np.argwhere(freq == freqs).ravel()
        w_f: np.ndarray = w[indices]
      else:
        w_f = w.copy()

      # Sort w_f by amplitude
      if len(result) == 0:
        i = np.argmax(self._calc_amp(w_f))
        result.append(w_f[i])
        w_f = np.delete(w_f, i, axis=0)

      while w_f.size > 0:
        corr = [np.correlate(seq, result[-1]) for seq in w_f]
        i = np.argmax(corr)
        result.append(w_f[i])
        w_f = np.delete(w_f, i, axis=0)

    return np.array(result)

  def scatter(self, ax: plt.Axes, xs, ys, xlabel='x-label', ylabel='y-label',
              title=None,):
    ax.scatter(xs, ys)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title: ax.set_title(title)

    # Set [xy]lim if required
    xmax = self.get('xmax')
    if xmax: ax.set_xlim(0, xmax)
    ymax = self.get('ymax')
    if ymax: ax.set_ylim(0, ymax)

    # Force integer ticks
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # TODO
    p = self.get('xper')
    if p:
      v = np.percentile(xs, p)
      max_y = max(ys) if not ymax else ymax
      ax.plot([v, v], [0, max_y], 'r-', alpha=0.5)
      if not title: ax.set_title(f'Percentile: {p}%')

  def imshow_w(self, ax: plt.Axes, w, fig):
    wmax = self.get('wmax')
    vmin, vmax = (-wmax, wmax) if wmax else (None, None)

    interp = None
    im = ax.imshow(
      w, cmap=self.get('cmap'), vmin=vmin, vmax=vmax,
      aspect='auto', interpolation=interp)

    # Add color bar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax)

  def sort_and_plot(self,fig: plt.Figure, x: np.ndarray):
    # Clear figure, create subplots
    fig.clear()
    axes = fig.subplots(1, 2, gridspec_kw={'width_ratios': [5, 3]})

    # (1) Create a scatter plot
    xs, ys = self.amplitude, self.sign_frequency
    self.scatter(axes[0], xs, ys, 'Amplitude', 'Frequency')

    # (2) Show weight as image
    if self.get('filter'):
      v = np.percentile(xs, self.get('xper'))
      x = x[np.argwhere(self.amplitude > v).ravel()]
    self.imshow_w(axes[1], self.sort_rows(x), fig)

  # endregion: Plotting Method



if __name__ == '__main__':
  # (1) Load note and get weight matrices
  sum_path = r'0404_mod_tasnetEMG_GAN.sum'
  note = Note.load(sum_path)[0]
  weight_list = note.tensor_dict['Exemplar 0']['Encoder Weights']
  weight_list = [w.T for w in weight_list]

  # metric_key = 'val RRMSE_temporal'

  # (2) Plot weights
  EncoderAnalyzer.plot(weight_list, title='Encoder Analyzer',
                       fig_size=(10, 10))

