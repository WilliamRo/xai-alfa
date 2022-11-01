from tframe.trainers import SmartTrainerHub
from tframe.configs.config_base import Flag

from roma import console

import re



class SLPConfig(SmartTrainerHub):

  channels = Flag.string('0,1;2', 'Channels to read from edfs', is_key=None)
  train_id = Flag.integer(0, 'Train id for incremental learning', is_key=None)


  @property
  def channel_list(self): return re.split(',|;', self.channels)

  @property
  def channel_num(self): return len(self.channel_list)

  @property
  def fusion_channels(self):
    return [s.split(',') for s in self.channels.split(';')]

  @property
  def is_all_data_scene(self):
    if self.data_config is None: return None
    return ',' not in self.data_config

# New hub class inherited from SmartTrainerHub must be registered
SLPConfig.register()

