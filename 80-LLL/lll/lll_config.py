import enum

from tframe.trainers import SmartTrainerHub
from tframe.configs.config_base import Flag

from roma import console



class LLLConfig(SmartTrainerHub):

  class Tasks(enum.Enum):
    MNIST = 'MNIST'
    FMNIST = 'FMNIST'

  task = Flag.string(None, 'Task', is_key=True)

  # region: Private Methods

  def sanity_check(self):
    assert self.task in self.Tasks.__members__

  # endregion: Private Methods





# New hub class inherited from SmartTrainerHub must be registered
LLLConfig.register()

