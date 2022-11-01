import os
from roma import finder


src_path = os.path.abspath('.')
dst_path = r'F:\XAI Dropbox\William Ro\03-William@XAI\80-Yau-Awards\Code\sleep_il'

ignored_patterns=('.*', '__*__', 'checkpoints', 'logs', 'tests', 'xai-kit',
                  '01-data', 'sync.py', 'mascot.py')
finder.synchronize(src_path, dst_path, pattern='*.py',
                   ignored_patterns=ignored_patterns, verbose=True)