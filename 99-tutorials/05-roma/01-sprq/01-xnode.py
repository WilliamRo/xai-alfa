from roma.spqr.threading import XNode
from pictor import Pictor

import time



class Box(Pictor, XNode):

  def __init__(self):
    super(Box, self).__init__()
    self.shortcuts.register_key_event('a', self.async_tick, 'async_tick')
    self.shortcuts.register_key_event('s', self.stop, 'stop async_tick')


  def async_tick(self):
    def _tick():
        if not hasattr(_tick, 'i'):
          _tick.i = 0
          _tick.id = len(self.child_nodes)
        _tick.i += 1
        time.sleep(1)
        print(f'Node # {_tick.id}: {_tick.i}')
    self.execute_a_new_child(_tick)


  def stop(self):
    self.terminate()


  def ls_children(self):
    for c in self.child_nodes: print(c)




b = Box()
b.show()




