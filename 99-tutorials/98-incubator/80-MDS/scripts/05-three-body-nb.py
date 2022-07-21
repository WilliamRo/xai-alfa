import numpy as np

from nanobox.box import NanoBox



DIM = 2
nb = NanoBox(3, f'Three Body ({DIM}D)')

init_r = np.array([[0., 2.], [2., 0.], [-1., 0.]])
nb.register_var('r', DIM, init_value=init_r)
nb.register_var('v', DIM)
nb.register_var('f', DIM)

def simulate(dt, track):
  pass
nb.simulate = simulate


nb.show()