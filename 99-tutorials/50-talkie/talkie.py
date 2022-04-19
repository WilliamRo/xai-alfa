import matplotlib.pyplot as plt

from pictor import DaVinci
import _thread, threading



class Talkie(DaVinci):

  MAX_CHARS = 40

  def __init__(self):
    # Call parent's constructor
    super(Talkie, self).__init__('Talkie', 5, 5)

    # Add plotter
    self.add_plotter(self.blah)

    # Private fields
    self._messages = []


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


  def add_msg(self, text):
    self._messages.append(text)
    self._draw()
  am = add_msg


  def go(self):
    def _go():
      import time
      self.add_msg('hahhahahhahhahhah')
      time.sleep(1)
      self.add_msg('hahhahahhahhahhah')
      time.sleep(1)
      self.add_msg('hahhahahhahhahhah')
    # _thread.start_new_thread(_go, ())
    t = threading.Thread(target=_go)
    t.start()



if __name__ == '__main__':
  tk = Talkie()
  tk.show()