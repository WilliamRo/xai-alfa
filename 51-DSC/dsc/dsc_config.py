from tframe.trainers import SmartTrainerHub
from tframe.configs.config_base import Flag

from roma import console


class DSConfig(SmartTrainerHub):

  class Datasets:
    pass


# New hub class inherited from SmartTrainerHub must be registered
DSConfig.register()

