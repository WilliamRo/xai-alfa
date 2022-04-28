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

    # Draw vertical lines
    line_i = [i for i in indices if i % 20 == 0]
    for i in line_i: ax.plot([i, i], [0, 1], 'r-')

    # Set axis limit
    ax.set_xlim([indices[0], indices[-1]])
    ax.set_ylim([-1, 2])




if __name__ == '__main__':
  sv = SeqViewer()
  sv.set_data(np.random.random(1000))
  sv.show()




