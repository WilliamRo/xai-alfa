import numpy as np

from nanobox.box import NanoBox



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
  new_r = r + new_v * dt

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
