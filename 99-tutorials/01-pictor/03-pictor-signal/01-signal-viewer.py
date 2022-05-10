import numpy as np

from pictor import Pictor
from pictor.objects import DigitalSignal


p = Pictor.signal_viewer(default_win_size=1000)

# Generate fake signals
x = np.linspace(0, 8*np.pi, num=1000)
p.objects = [DigitalSignal.sinusoidal(x, omega=[4, 8, 8, 12, 4, 4],
                                      max_truncate_ratio=0.5)]

p.show()





