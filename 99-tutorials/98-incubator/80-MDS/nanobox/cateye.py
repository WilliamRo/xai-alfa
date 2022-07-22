from pictor.objects.particle_dynamic import ParticleSystem
from pictor.plotters.plotter_base import Plotter
from pictor.plugins.timer import Timer
from roma.spqr.threading import XNode

import matplotlib.pyplot as plt



class CatEye(Plotter, XNode, Timer):

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

  def plot_particles(self, x: tuple, ax: plt.Axes):
    """Objects in NanoBox should be a list of tuples (track_name, t)"""
    # If x is not provided
    if x is None:
      self.show_text('No particle found', ax=ax)
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

  # region: Registration

  def _register_settable_attributes(self):
    self.new_settable_attr('xmin', None, float, 'x-axis minimal value')
    self.new_settable_attr('xmax', None, float, 'x-axis maximum value')
    self.new_settable_attr('ymin', None, float, 'y-axis minimal value')
    self.new_settable_attr('ymax', None, float, 'y-axis maximum value')
    self.new_settable_attr('xlabel', None, str, 'Label of x-axis')
    self.new_settable_attr('ylabel', None, str, 'Label of y-axis')
    self.new_settable_attr('size', 1000, float, 'Particle size')

  def register_shortcuts(self):
    self.register_a_shortcut('space', self.play_pause, 'Play')

  # endregion: Registration

  # region: Threading

  def play_pause(self):
    if self.child_nodes:
      print('Pause')
      self.child_nodes[0].terminate()
      self.pictor.title = self.pictor.static_title
    else:
      print('Play')
      self.execute_a_new_child(self._play)

  def _play(self):
    self._tic()
    # Press 'j'
    self.pictor.shortcuts.library['j'][0]()
    self.pictor.title = f'{self.pictor.static_title} (FPS = {self.fps:.1f})'
    self.refresh(in_thread=True)

  # endregion: Threading
