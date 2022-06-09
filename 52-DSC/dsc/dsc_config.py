from tframe.trainers import SmartTrainerHub
from tframe.configs.config_base import Flag

from roma import console


class DSConfig(SmartTrainerHub):

  class Datasets:
    RML2016 = 'rml2016'

  report_detail = Flag.boolean(False, 'Whether to report detail')


# New hub class inherited from SmartTrainerHub must be registered
DSConfig.register()

