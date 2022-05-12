"""-f pictor.plotters"""
from pictor.plotters import Plotter

import matplotlib.pyplot as plt
import numpy as np
from pictor_sleep.sleep_data import SleepData



class Monitor(Plotter):

  def __init__(self, pictor=None):
    # Call parent's constructor
    super(Monitor, self).__init__(self.show_data, pictor)


  def show_data(self, x: SleepData, fig: plt.Figure, i: int):
    ax: plt.Axes = fig.add_subplot(211)
    ax.text(0.5, 0.5, f'Hello World {x}', ha='center', va='center')
    ax.axis('off')

    ax2: plt.Axes = fig.add_subplot(212)
    ax2.plot(np.sin(np.linspace(-4, 4, 1000)))



  def register_shortcuts(self):
    self.register_a_shortcut('h', lambda: print('hahhahahah'),
                             description='Laugh')

