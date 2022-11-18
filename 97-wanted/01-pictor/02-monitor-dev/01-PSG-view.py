from pictor import Pictor
from slp.slp_datasets.ucddb import UCDDB
from slp_core import th

from monitor_pro import MonitorPro



th.data_config = 'ucddb'
data_set = UCDDB.load_as_tframe_data(th.data_dir)

p = Pictor(title='UCDDB-1.0.0', figure_size=(12, 8))
p.objects = data_set.signal_groups

m: MonitorPro = p.add_plotter(MonitorPro(channels='C3A2,C4A1'))
p.show()

