from roma.spqr.threading import XNode
from pictor import Pictor
from pictor.plugins.timer import Timer
from pictor.objects import ParticleSystem



class NanoBox(Pictor, XNode, Timer):
  pass



if __name__ == '__main__':
  na = NanoBox()
  na.show()
