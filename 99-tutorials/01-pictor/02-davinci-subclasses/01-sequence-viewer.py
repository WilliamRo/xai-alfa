from pictor import DaVinci

import matplotlib.pyplot as plt
import numpy as np



class SeqViewer(DaVinci):

  WIN_SIZE = 100
  OVERLAP = 10

  def __init__(self):
    super(SeqViewer, self).__init__('Sequence Viewer')
    self.data = None
    self.add_plotter(self.show_window)


  def set_data(self, data):
    assert isinstance(data, np.ndarray) and len(data.shape) == 1
    self.data = data
    self.objects = np.arange(0, len(data) - self.WIN_SIZE, self.OVERLAP)


  def show_window(self, x, ax: plt.Axes):
    indices = np.arange(x, x + self.WIN_SIZE)
    ax.plot(indices, self.data[indices])



if __name__ == '__main__':
  sv = SeqViewer()
  sv.set_data(np.random.random(1000))
  sv.show()




