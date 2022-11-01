"""
指定原始数据路径，封装成tframe.DataSet子类，并用pictor可视化
"""
from slp.slp_agent import SLPAgent
from slp.slp_set import SleepSet


# Specify path
file_path = r'E:\xai-alfa\51-SLEEP\data\sleepedf-lll'

# Read data
data_set = SLPAgent.load_as_tframe_data(file_path)
assert isinstance(data_set, SleepSet)

# Visualize using pictor and monitor
data_set.show()

