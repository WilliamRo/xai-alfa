import sys, os
FILE_PATH = os.path.abspath(__file__)
ROOT = FILE_PATH
# Specify the directory depth with respect to the root of your project here
# (The project root usually holds your data folder and has a depth of 0)
DIR_DEPTH = 0
for _ in range(DIR_DEPTH + 1):
  ROOT = os.path.dirname(ROOT)
  sys.path.insert(0, ROOT)

sys.path.insert(0, os.path.join(ROOT, '80-LLL'))

from tframe.utils.summary_viewer import main_frame
from tframe import local



current_dir = os.path.dirname(FILE_PATH)
if len(sys.argv) == 2:
  path = sys.argv[1]
  if os.path.exists(path):
    current_dir = path
current_dir = os.path.join(current_dir, '80-LLL')


while True:
  try:
    summ_path = local.wizard(extension='sum', max_depth=3,
                             current_dir=current_dir,
                             input_with_enter=len(sys.argv) == 1)
                             # input_with_enter=False)

    if summ_path is None:
      input()
      continue
    print('>> Loading notes, please wait ...')

    from pictor import Pictor
    from lll_studio.bamboo import Bamboo

    p = Pictor(figure_size=(10, 5))
    bb = Bamboo(p)
    plotter = p.add_plotter(bb)
    bb.load_notes(summ_path)
    p.show()

  except Exception as e:
    import sys, traceback
    traceback.print_exc(file=sys.stdout)
    input('Press any key to quit ...')
    raise e
  print('-' * 80)
