from pictor import Pictor




class Omma(Pictor):

  def _register_default_key_events(self):
    super(Omma, self)._register_default_key_events()

    # Remove close key
    self.shortcuts.library.pop('q')
    self.shortcuts.library.pop('Escape')
    self.shortcuts.register_key_event('Q', self.quit, 'Quit')

    # Use command `q` to quit
    self.q = self.quit



if __name__ == '__main__':
  om = Omma()
  om.show()
