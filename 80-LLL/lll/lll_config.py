import enum

from tframe.trainers import SmartTrainerHub
from tframe.configs.config_base import Flag

from roma import console



class LLLConfig(SmartTrainerHub):

  class Tasks(enum.Enum):
    MNIST = 'MNIST'
    FMNIST = 'FMNIST'
    SLEEPEDF = 'SLEEPEDF'

  task = Flag.string(None, 'Task', is_key=True)
  train_id = Flag.integer(None, 'Who is training?', is_key=True)
  val_pct = Flag.float(0.1, '...', is_key=None)

  trial_id = Flag.integer(0, '...', is_key=None)

  # region: Private Methods

  def sanity_check(self):
    if not isinstance(self.task, str):
      self.task = self.task.value

  # endregion: Private Methods





# New hub class inherited from SmartTrainerHub must be registered
LLLConfig.register()

