from pictor.objects.particle_dynamic import ParticleSystem
from pictor.plotters.plotter_base import Plotter

import matplotlib.pyplot as plt
import numpy as np



class CatEye(Plotter):

  def __init__(self, pictor=None):
    # Call parent's constructor
    super(CatEye, self).__init__(self.plot_particles, pictor)

    #
    self._register_settable_attributes()

  # region: Properties

  @property
  def system(self) -> ParticleSystem: return self.pictor

  # endregion: Properties

  # region: Plot Methods

  def plot_particles(self, x: tuple, ax: plt.Axes, fig: plt.Figure):
    """Objects in NanoBox should be a list of tuples (track_name, t)"""
    # If x is not provided
    if x is None:
      self.show_text('No particle found', fig=fig)
      return

    # Unpack x
    track, t = x

    # Get necessary values
    coords, colors = self.system.get_values(t, 'coords', 'colors', track=track)
    assert coords is not None

    # Plot
    ax.scatter(coords[:, 0], coords[:, 1], marker='o', c=colors,
               s=self.get('size'))

    # Set styles
    ax.set_xlim(self.get('xmin'), self.get('xmax'))
    ax.set_ylim(self.get('ymin'), self.get('ymax'))
    ax.set_xlabel(self.get('xlabel'))
    ax.set_ylabel(self.get('ylabel'))

  # endregion: Plot Methods

  # region: Private Methods

  def _register_settable_attributes(self):
    self.new_settable_attr('xmin', None, float, 'x-axis minimal value')
    self.new_settable_attr('xmax', None, float, 'x-axis maximum value')
    self.new_settable_attr('ymin', None, float, 'y-axis minimal value')
    self.new_settable_attr('ymax', None, float, 'y-axis maximum value')
    self.new_settable_attr('xlabel', None, str, 'Label of x-axis')
    self.new_settable_attr('ylabel', None, str, 'Label of y-axis')
    self.new_settable_attr('size', 1000, float, 'Particle size')

  # endregion: Private Methods
