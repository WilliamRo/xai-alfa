from tframe.trainers import SmartTrainerHub
from tframe.configs.config_base import Flag

from roma import console


class SLPConfig(SmartTrainerHub):

  class Datasets:
    ucddb = 'ucddb'

  report_detail = Flag.boolean(False, 'Whether to report detail')


# New hub class inherited from SmartTrainerHub must be registered
SLPConfig.register()

