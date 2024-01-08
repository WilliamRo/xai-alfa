from pictor import Pictor
from pictor.plotters import Plotter

import matplotlib.pyplot as plt
import numpy as np



t = np.linspace(-5, 5, 100)
sint = np.sin(t)

class SinePlotter(Plotter):
  def __init__(self, pictor=None):
    # Call parent's constructor
    super(SinePlotter, self).__init__(self.plot_sine, pictor)

    self.new_settable_attr('color', 'r', str, 'Color ...')

  def plot_sine(self, ax: plt.Axes, x):
    ax.plot(t, sint)
    y = np.sin(x)
    style = self.get('color') + 's'
    ax.plot(x, y, style)
    ax.set_title(f'({x}, {y})')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')

  def change_to_cosine(self):
    """Change sine to cosine"""
    for i in range(len(sint)): sint[i] = np.cos(t[i])
    self.refresh()
  ctc = change_to_cosine


p = Pictor(title='Sine Curve', figure_size=[8, 4])
p.objects = t
p.add_plotter(SinePlotter())

p.show()
