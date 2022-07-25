import numpy as np

from nanobox.box import NanoBox



init_coords = np.array([[0., 2., -1.],
                        [2., 0., 0.],
                        [-1., 0., 1.]])
dim = len(init_coords[0])

nb = NanoBox(len(init_coords), f'Three Body ({dim}D)', dimension=dim)

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
R = 3.5
nb.config_plotter(xmin=-R, xmax=R, ymin=-R, ymax=R, zmin=-R, zmax=R,
                  xlabel='meters', ylabel='meters', zlabel='meters', size=500)

nb.calculate_steps(0.05, 199, auto_refresh=False)
nb.show()
