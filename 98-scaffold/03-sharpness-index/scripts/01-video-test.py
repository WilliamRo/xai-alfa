from pictor import Pictor
from sharp_estimator import SharpEstimator



# -----------------------------------------------------------------------------
# Import your images here
# -----------------------------------------------------------------------------
images = []


# -----------------------------------------------------------------------------
# Show in Pictor
# -----------------------------------------------------------------------------
p = Pictor(title='Sharpness Estimator')

p.add_plotter(SharpEstimator(mode='image'))
p.add_plotter(SharpEstimator(mode='spectrum'))
p.add_plotter(SharpEstimator(mode='curve'))

p.objects = images

p.show()
