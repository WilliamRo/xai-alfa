from pictor import Pictor

import matplotlib.pyplot as plt
import numpy as np



t = np.linspace(-5, 5, 100)
sint = np.sin(t)

def plot_sine(ax: plt.Axes, x):
  ax.plot(t, sint)
  y = np.sin(x)
  ax.plot(x, y, 'rs')
  ax.set_title(f'({x}, {y})')
  ax.set_xlabel('Time')
  ax.set_ylabel('Amplitude')


p = Pictor(title='Sine Curve', figure_size=[8, 4])
p.objects = t
p.add_plotter(plot_sine)

p.show()
