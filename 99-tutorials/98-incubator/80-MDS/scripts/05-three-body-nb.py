import numpy as np

from nanobox.box import NanoBox



init_coords = np.array([[0., 2.], [2., 0.], [-1., 0.]])
dim = len(init_coords[0])

nb = NanoBox(len(init_coords), f'Three Body ({dim}D)')

nb.register_var('coords', dim, init_value=init_coords)
nb.register_var('v', dim)
nb.register_constants(colors=['b', 'k', 'r'], m=1., r0=1., ks=5.)


def simulate(pkg: dict, dt: float):
  r, r0, ks, v, m = [pkg[k] for k in ('coords', 'r0', 'ks', 'v', 'm')]
  f = np.zeros_like(r)

  # Compute force
  for i in range(3):
    for j in range(3):
      if i != j:
        rij = r[i] - r[j]
        rij_abs = np.linalg.norm(rij)
        f[i] -= ks * (rij_abs - r0) * rij / rij_abs

  # Update coords
  new_v = v + f / m * dt
  new_r = r + new_v * dt
  return {'coords': new_r, 'v': new_v}


nb.simulate = simulate
nb.set_plotter(xmin=-3.5, xmax=3.5, ymin=-3.5, ymax=3.5,
               xlabel='meters', ylabel='meters')

nb.calculate_steps(0.05, 199, auto_refresh=False)
nb.show()
