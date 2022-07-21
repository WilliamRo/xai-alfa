from tframe.configs.config_base import Flag
from tframe.trainers import SmartTrainerHub


class CASPConfig(SmartTrainerHub):
  pass


# New hub class inherited from SmartTrainerHub must be registered
CASPConfig.register()