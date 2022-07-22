from roma import console
from roma.spqr.threading import XNode
from pictor import Pictor
from pictor.plugins.timer import Timer
from pictor.objects import ParticleSystem
from nanobox.cateye import CatEye, Plotter



class NanoBox(Pictor, ParticleSystem, XNode, Timer):
  """A NanoBox is bound to a ParticleSystem
  TODO: <3-D> <play>
  """

  def __init__(self, num_particles, title='NanoBox', figure_size=(6, 6),
               plotter=None):
    # Call parents' constructors
    Pictor.__init__(self, title, figure_size)
    ParticleSystem.__init__(self, num_particles)

    # Set plotter
    if plotter is None: plotter = CatEye(self)
    self.add_plotter(plotter)

  # region: Properties

  # endregion: Properties

  # region: Public Methods

  def refresh_objects(self, track=ParticleSystem.DEFAULT_TRACK):
    self.objects = [(track, t) for t in self.timelines[track].keys()]

  def calculate_steps(self, dt: float, num_steps: int,
                      track: str = ParticleSystem.DEFAULT_TRACK,
                      auto_refresh=True):
    for i in range(num_steps):
      console.print_progress(i, num_steps)
      self._move_forward(dt, track)

    console.show_status(f'Calculated {num_steps} steps.')
    self.refresh_objects()
    if auto_refresh: self.refresh()
  cs = calculate_steps

  # endregion: Public Methods



if __name__ == '__main__':
  nb = NanoBox(3)
  nb.show()
