from pictor import DaVinci

import numpy as np
import matplotlib.pyplot as plt



da = DaVinci(height=5, width=10)
da.objects = np.arange(-2, 2, 0.05)


def plotter(x):
  a = np.arange(-2, 2, 0.05)

  plt.subplot(1, 2, 1)
  plt.plot(a, np.sin(a))
  plt.plot(x, np.sin(x), 'rs')

  plt.subplot(1, 2, 2)
  plt.plot(a, np.cos(a))
  plt.plot(x, np.cos(x), 'rs')

da.add_plotter(plotter)
da.show()
