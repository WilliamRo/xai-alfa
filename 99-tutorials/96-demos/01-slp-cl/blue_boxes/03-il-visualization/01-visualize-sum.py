from bamboo import Bamboo
from pictor import Pictor



summ_path = r'E:\archive\sleep_il\01-SLEEP\03_il_data_fusion\1029_s1_il_data_fusion.sum'

Bamboo.N_SPLITS = 3

p = Pictor(figure_size=(7, 3))
bb = Bamboo(p)
plotter = p.add_plotter(bb)
bb.load_notes(summ_path)
p.show()
