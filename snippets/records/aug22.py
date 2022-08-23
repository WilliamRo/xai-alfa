from pictor import Pictor

import numpy as np
import matplotlib.pyplot as plt


# Button
pieces = [
  91.6, 91.3, 114.9, 114.8, 115.6, 115.4
]

def plotter(ax: plt.Axes):
  ticks = list(range(len(pieces)))
  ax.plot(ticks, pieces)
  ax.set_xticks(ticks)
  ax.set_xlabel('Rounds (x7)')
  ax.set_ylabel('Pieces (K)')

p = Pictor(figure_size=(7, 3))
p.add_plotter(plotter)
p.show()
