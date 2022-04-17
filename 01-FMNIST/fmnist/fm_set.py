from pictor import DaVinci
from tframe import DataSet
from tframe import pedia



class FMSet(DataSet):

  def show(self):
    da = DaVinci(f'FMNIST - {self.name}', init_as_image_viewer=True)
    da.objects = self.features
    da.object_titles = [self.properties[pedia.classes][c]
                        for c in self.dense_labels]
    da.show()