from fmnist.fm_agent import FMNIST



def load_data():
  from fm_core import th
  return FMNIST.load(th.data_dir)



if __name__ == '__main__':
  train_set, val_set, test_set = load_data()
  train_set.show()

