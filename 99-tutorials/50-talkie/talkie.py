import matplotlib.pyplot as plt

from pictor import DaVinci
from threading import Thread
from typing import Optional


class Talkie(DaVinci):

  MAX_CHARS = 40

  def __init__(self):
    # Call parent's constructor
    super(Talkie, self).__init__('Talkie', 5, 5)

    # Add plotter
    self.add_plotter(self.blah)

    # Private fields
    self._messages = []
    self._listener: Optional[Thread] = None


  def blah(self, ax: plt.Axes):
    # Show status
    s1 = 'Status 1: --'
    ax.text(0, 1, s1, ha='left', va='top')
    s2 = 'Status 2: --'
    ax.text(0, 0.95, s2, ha='left', va='top')

    # Show messages
    msg = '\n\n'.join(['Messages\n' + '-' * 15] + [
      m[:self.MAX_CHARS] for m in self._messages[-5:]])
    ax.text(0.5, 0.8, msg, ha='center', va='top', wrap=True)

    # Hide the axis
    ax.axis('off')


  def add_msg(self, text, in_thread: bool = True):
    self._messages.append(text)
    self._draw(in_thread=in_thread)
  am = add_msg


  def go(self):
    import time
    def _go():
      self.add_msg('gogogogogogo', in_thread=True)
      for _ in range(10):
        time.sleep(2)
        self.add_msg('gogogogogogo', in_thread=True)

    self._listener = Thread(target=_go)
    self._listener.start()


  # region: Overwriting

  def __del__(self):
    if isinstance(self._listener, Thread) and self._listener.is_alive():
      pass
    print('deling')

  # endregion



if __name__ == '__main__':
  tk = Talkie()
  tk.show()