from typing import Optional
from pictor import DaVinci

import matplotlib.pyplot as plt
import numpy as np
import threading
import time


class NanoBox(DaVinci):

  def __init__(self):
    super(NanoBox, self).__init__('Three Body', 6, 6)
    self._static_title = self.title

    # Parameters of the problem
    self.T = 10.   # s
    self.m = 1.0   # kg
    self.ks = 5    # N/m
    self.r0 = 1.   # m

    # Setting a time-step to be 50 ms
    self.dt = 0.05  # s
    self.N = int(self.T / self.dt)

    # Allocating arrays for 2D problem: first axis - time. second axis -
    #   particle's number. third - coordinate
    self.v = np.zeros((self.N + 1, 3, 2))
    self.r = np.zeros((self.N + 1, 3, 2))
    self.f = np.zeros((self.N + 1, 3, 2))

    # Initial conditions for 3 particles:
    self.r[0, 0] = np.array([0., 2.])
    self.r[0, 1] = np.array([2., 0.])
    self.r[0, 2] = np.array([-1., 0.])

    # Run dynamics:
    for n in range(self.N):
      self.compute_forces(n)
      self.v[n + 1] = self.v[n] + self.f[n] / self.m * self.dt
      self.r[n + 1] = self.r[n] + self.v[n + 1] * self.dt

    # Set objects and plotter
    self.objects = range(int(self.T / self.dt))
    self.add_plotter(self.player)

    # Bind keys
    self.state_machine.register_key_event(' ', self.play)
    self.state_machine.library['h'] = lambda: self.tune_fps(-1)
    self.state_machine.library['l'] = lambda: self.tune_fps(1)
    self.state_machine.library['H'] = lambda: self.tune_fps(-10)
    self.state_machine.library['L'] = lambda: self.tune_fps(10)

    # Player variables
    self.t: Optional[threading.Thread] = None
    self.fps = 10
    self._tic = None


  def compute_forces(self, n):
    '''The function computes forces on each pearticle at time step n'''
    for i in range(3):
      for j in range(3):
        if i != j:
          rij = self.r[n, i] - self.r[n, j]
          rij_abs = np.linalg.norm(rij)
          self.f[n, i] -= self.ks * (rij_abs - self.r0) * rij / rij_abs


  def player(self, x, ax: plt.Axes):
    ax.scatter(self.r[x, :, 0], self.r[x, :, 1], marker='o', c=['b', 'k', 'r'],
               s=1000)

    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_xlabel('meters')
    ax.set_ylabel('meters')


  def play(self):
    def _play():
      _t = self.t
      while not _t.should_stop:
        self._object_cursor = (self._object_cursor + 1) % len(self.objects)

        # Show fps
        tic = time.time()
        if self._tic is not None:
          self.title = '{} (FPS = {:.1f} < {})'.format(
            self._static_title, 1 / (tic - self._tic), self.fps)
        self._tic = tic

        self._draw(in_thread=True)
        time.sleep(1 / self.fps)
      print('Thread terminated.')

    if not isinstance(self.t, threading.Thread) or not self.t.is_alive():
      self.t = threading.Thread(target=_play)
      self.t.should_stop = False
      self.t.start()
    else:
      self.t.should_stop = True
      self.title = self._static_title
      self._tic = None
      self.refresh()


  def tune_fps(self, delta: int):
    if delta + self.fps < 1: return
    self.fps = self.fps + delta



if __name__ == '__main__':
  na = NanoBox()
  na.show()
