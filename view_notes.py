import sys, os
FILE_PATH = os.path.abspath(__file__)
ROOT = FILE_PATH
# Specify the directory depth with respect to the root of your project here
# (The project root usually holds your data folder and has a depth of 0)
DIR_DEPTH = 1
for _ in range(DIR_DEPTH + 1):
  ROOT = os.path.dirname(ROOT)
  sys.path.insert(0, ROOT)
from tframe.utils.summary_viewer import main_frame
from tframe import local

from tframe.utils.tensor_viewer.plugins import distribution


current_dir = os.path.dirname(FILE_PATH)
if len(sys.argv) == 2:
  path = sys.argv[1]
  if os.path.exists(path):
    current_dir = path


default_inactive_flags = (
  'epoch',
  'early_stop',
  'mark',
  'total_params',
)

default_inactive_criteria = (
  'Best Accuracy',
  'Best F1',
)

flags_to_ignore = (
  'mark',
  'warm_up_thres',
  'epoch',
  'early_stop',
  'optimizer',
  'learning_rate',
)

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
    viewer = main_frame.SummaryViewer(
      summ_path,
      default_inactive_flags=default_inactive_flags,
      default_inactive_criteria=default_inactive_criteria,
      flags_to_ignore=flags_to_ignore,
    )
    viewer.register_plugin(distribution.plugin)
    viewer.show()

  except Exception as e:
    import sys, traceback
    traceback.print_exc(file=sys.stdout)
    input('Press any key to quit ...')
    raise e
  print('-' * 80)
