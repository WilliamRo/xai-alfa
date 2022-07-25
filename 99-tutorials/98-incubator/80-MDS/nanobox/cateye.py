import time

from mpl_toolkits.mplot3d import Axes3D
from pictor.objects.particle_dynamic import ParticleSystem
from pictor.plotters.plotter_base import Plotter
from pictor.plugins.timer import Timer
from roma.spqr.threading import XNode
from typing import Union

import matplotlib.pyplot as plt



class CatEye(Plotter, XNode, Timer):

  def __init__(self, pictor=None, dimension=2):
    # Call parent's constructor
    super(CatEye, self).__init__(self.get_plotter(dimension), pictor)

    #
    self._register_settable_attributes()

  # region: Properties

  @property
  def system(self) -> ParticleSystem: return self.pictor

  # endregion: Properties

  # region: Plot Methods

  def get_plotter(self, dim):
    if dim == 2: return self.plot_particles
    return lambda x, ax3d: self.plot_particles(x, ax3d)

  def plot_particles(self, x: tuple, ax: Union[plt.Axes, Axes3D]):
    """Objects in NanoBox should be a list of tuples (track_name, t)"""
    plot_3D = isinstance(ax, Axes3D)

    # If x is not provided
    if x is None:
      if plot_3D: ax = self.pictor.canvas.axes2D
      self.show_text('No particle found', ax=ax)
      return

    # Unpack x
    track, t = x

    # Get necessary values
    coords, colors = self.system.get_values(t, 'coords', 'colors', track=track)
    assert coords is not None

    # Plot
    ax.scatter(*[coords[:, i] for i in range(coords.shape[1])],
               marker='o', c=colors, s=self.get('size'))

    # Set styles
    ax.set_xlim(self.get('xmin'), self.get('xmax'))
    ax.set_ylim(self.get('ymin'), self.get('ymax'))
    ax.set_xlabel(self.get('xlabel'))
    ax.set_ylabel(self.get('ylabel'))

    if plot_3D:
      ax.set_zlim(self.get('zmin'), self.get('zmax'))
      ax.set_zlabel(self.get('zlabel'))
      ax.view_init(*self.pictor.canvas.view_angle)

  # endregion: Plot Methods

  # region: Registration

  def _register_settable_attributes(self):
    self.new_settable_attr('xmin', None, float, 'x-axis minimal value')
    self.new_settable_attr('xmax', None, float, 'x-axis maximum value')
    self.new_settable_attr('ymin', None, float, 'y-axis minimal value')
    self.new_settable_attr('ymax', None, float, 'y-axis maximum value')
    self.new_settable_attr('zmin', None, float, 'z-axis minimal value')
    self.new_settable_attr('zmax', None, float, 'z-axis maximum value')
    self.new_settable_attr('xlabel', None, str, 'Label of x-axis')
    self.new_settable_attr('ylabel', None, str, 'Label of y-axis')
    self.new_settable_attr('zlabel', None, str, 'Label of z-axis')
    self.new_settable_attr('size', 1, float, 'Particle size')

  def register_shortcuts(self):
    self.register_a_shortcut('space', self.play_pause, 'Play')

  # endregion: Registration

  # region: Threading

  def play_pause(self):
    if self.child_nodes:
      self.child_nodes[0].terminate()
      # Enable refreshing from main thread
      self.pictor.allow_main_thread_refreshing(True)
    else:
      # Disable refreshing from main thread to avoid conflicts
      self.pictor.allow_main_thread_refreshing(False)
      self.execute_a_new_child(self._play, daemon=True)

  def _play(self):
    box = self.pictor
    self._tic()
    # Press 'j'
    box.set_cursor(box.Keys.OBJECTS, 1, refresh=False)
    # After termination, this title suffix will still exist
    box.title_suffix = f' (FPS = {self.fps:.1f})'
    self.refresh(wait_for_idle=True)

  # endregion: Threading
