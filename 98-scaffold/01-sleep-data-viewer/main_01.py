from pictor import Pictor
from pictor_sleep.monitor import Monitor
from pictor_sleep.sleep_data import SleepData



# Load data
# ...


# Initiate a pictor
p = Pictor(title='Sleep Monitor', figure_size=(5, 5))

# Set plotter
m = Monitor()
p.add_plotter(m)

# Set objects
p.objects = [99, 88, 777]

# Begin main loop
p.show()







