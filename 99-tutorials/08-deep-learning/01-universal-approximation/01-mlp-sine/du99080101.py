from tframe import DataSet

import numpy as np



class DiscreteSine(DataSet):

  f = np.sin

  @classmethod
  def get_data(cls, vmin=-5, vmax=5, num=50):
    x = np.linspace(vmin, vmax, num)
    y = cls.f(x)
    return DiscreteSine(x, y, name=f'Sine([{vmin}, {vmax}])N{num}')

  def visualize(self):
    from pictor import Pictor
    p = Pictor(self.name, figure_size=(8, 5))
    p.objects = [(self.features, self.targets)]
    p.add_plotter(lambda x, ax: ax.plot(x[0], x[1]))
    p.add_plotter(lambda x, ax: ax.scatter(x[0], x[1]))
    p.show()



if __name__ == '__main__':
  data = DiscreteSine.get_data(num=50)
  data.visualize()
