"""
配置（1）病人分组 （2）数据增强方法  （3）通道选择
后报告数据格式
"""
from slp_agent import SLPAgent
from slp_core import th
from roma import console


# Specify path
file_path = r'E:\xai-alfa\51-SLEEP\data\sleepedf-lll'

# (1) Generate patient groups
th.data_config = ['7', '3,2,2'][0]
# (2) Data augmentation
th.aug_config = ['', '2*3', '2*3+4*3'][1]
# (3) Channel selection
th.channels = '0,1;2'

# Read data (train 70%, val 10%, test 20%)
# ALL: data_sets = [[train1, val1, test1]]
# LLL: data_sets = [[train1, val1, test1], ..., [trainN, valN, testN]]
data_sets = SLPAgent.load(file_path)

# Report data details
for i, (train_set, val_set, test_set) in enumerate(data_sets):
  console.show_status(f'Group {i+1}:')
  train_set.report()
  val_set.report()
  test_set.report()


