from slp.slp_agent import SLPAgent



def load_data():
  from slp_core import th
  data_sets = SLPAgent.load(th.data_dir)
  return data_sets



if __name__ == '__main__':
  pass