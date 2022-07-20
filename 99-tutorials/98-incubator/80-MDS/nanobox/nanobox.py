from roma.spqr.threading import XNode
from pictor import Pictor
from pictor.plugins.timer import Timer
from pictor.objects import ParticleSystem

import matplotlib.pyplot as plt



class NanoBox(Pictor, XNode, Timer):

  def __init__(self, title='NanoBox', figure_size=(6, 6)):
    # Call parent's constructor
    super(NanoBox, self).__init__(title, figure_size)

    # Set plotter
    self.add_plotter(self.show_particles)


  # region: Properties

  # endregion: Properties

  # region: Plotters

  def show_particles(self, x: ParticleSystem, ax: plt.Axes):
    ax.set_title(f'x = {x}')

  # endregion: Plotters




if __name__ == '__main__':
  na = NanoBox()
  na.show()
