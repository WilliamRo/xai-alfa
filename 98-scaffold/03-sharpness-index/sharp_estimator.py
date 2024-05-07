from pictor import Pictor
from pictor import Plotter


import matplotlib.pyplot as plt
import numpy as np



class SharpEstimator(Plotter):

  def __init__(self, mode='image'):
    super(SharpEstimator, self).__init__(self.plot)

    assert mode in ('image', 'spectrum', 'curve')
    self.mode = mode

    self.new_settable_attr('log', False, bool, 'Use log-scale in k-space')


  def plot(self, x, ax: plt.Axes):
    if self.mode == 'image':
      ax.imshow(x)
      ax.set_axis_off()
    elif self.mode == 'spectrum':
      s = np.abs(np.fft.fft2(x))
      if self.get('log'): s = np.log(s + 1e-10)
      ax.imshow(s)
    elif self.mode == 'curve':
      xs, ys = self.calculate_sharpness_curve(x)
      ax.plot(xs, ys)
      ax.set_title(f'AUC = 0.00')
    else: raise NotImplemented


  def calculate_sharpness_curve(self, x):
    x = np.arange(20) / 20
    return x, x


  def register_shortcuts(self):
    self.register_a_shortcut(
      'L', lambda: self.flip('log'),
      'Turn on/off log scale in k-space view')



if __name__ == '__main__':
  p = Pictor(title='Sharpness Estimator')
  p.add_plotter(SharpEstimator(mode='image'))
  p.add_plotter(SharpEstimator(mode='spectrum'))
  p.add_plotter(SharpEstimator(mode='curve'))

  N = 100
  p.objects = [np.random.rand(N, N) for _ in range(10)]

  p.show()
