from scipy import stats

import numpy as np



repeats = 100
sig = 0.05
print(f'>> Testing normality, n_repeat={repeats}')
print('-' * 79)
for N in (10, 20, 50, 100, 200):
  for s1, s2 in ((1., 1.,), (1., 1.1), (1, 1.5)):
    acc = np.average(
      [p > sig for p in [
        stats.levene(np.random.randn(N) * s1,
                     np.random.randn(N) * s2,
                     center='mean')[1] for _ in range(repeats)]])
    if s1 != s2: acc = 1. - acc
    print(f'N = {N}, ({s1} & {s2}), Accuracy = {acc * 100:.1f}%')

  print('-' * 79)
