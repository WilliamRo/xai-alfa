from tframe.trainers import SmartTrainerHub
from tframe.configs.config_base import Flag

from roma import console



class LLLConfig(SmartTrainerHub):

  class DataSets:
    MNIST = 'mnist'
    FMNIST = 'fmnist'


# New hub class inherited from SmartTrainerHub must be registered
LLLConfig.register()

