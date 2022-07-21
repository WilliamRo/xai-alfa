from roma import console
from roma.spqr.threading import XNode
from pictor import Pictor
from pictor.plugins.timer import Timer
from pictor.objects import ParticleSystem
from nanobox.cateye import CatEye, Plotter

import numpy as np



class NanoBox(Pictor, ParticleSystem, XNode, Timer):
  """A NanoBox is bound to a ParticleSystem"""

  def __init__(self, num_particles, title='NanoBox', figure_size=(6, 6),
               plotter=None):
    # Call parents' constructors
    Pictor.__init__(self, title, figure_size)
    ParticleSystem.__init__(self, num_particles)

    # Set plotter
    if plotter is None: plotter = CatEye(self)
    self.add_plotter(plotter)

  # region: Properties

  @property
  def default_plotter(self) -> Plotter:
    return self.axes[self.Keys.PLOTTERS][0]

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
  init_coords = np.array([[0., 2.], [2., 0.], [-1., 0.]])
  dim = len(init_coords[0])
  nb = NanoBox(3, f'Three Body ({dim}D)')

  nb.register_var('coords', dim, init_value=init_coords)
  nb.register_var('v', dim)
  nb.register_var('f', dim)

  # Set constants
  nb.CONSTANTS['colors'] = ['b', 'k', 'r']
  nb.CONSTANTS['m'] = 1.0
  nb.CONSTANTS['r0'] = 1.0
  nb.CONSTANTS['ks'] = 5.

  def simulate(pkg: dict, dt: float):
    r, r0, f, ks, v, m = [pkg[k] for k in ('coords', 'r0', 'f', 'ks', 'v', 'm')]

    # Compute force
    for i in range(3):
      for j in range(3):
        if i != j:
          rij = r[i] - r[j]
          rij_abs = np.linalg.norm(rij)
          f[i] -= ks * (rij_abs - r0) * rij / rij_abs

    # Update coords
    new_f = np.zeros_like(f)
    new_v = v + f / m * dt
    new_r = r + v * dt

    return {'coords': new_r, 'v': new_v, 'f': new_f}

  nb.simulate = simulate

  nb.default_plotter.set('xmin', -3.5)
  nb.default_plotter.set('xmax', 3.5)
  nb.default_plotter.set('ymin', -3.5)
  nb.default_plotter.set('ymax', 3.5)
  nb.default_plotter.set('xlabel', 'meters')
  nb.default_plotter.set('ylabel', 'meters')

  nb.calculate_steps(0.05, 199, auto_refresh=False)
  nb.show()
