from dsc.dsc_set import DSCSet
from dsc.dsc_agent import DSCAgent



def load():
  from dsc_core import th

  train_set, val_set, test_set = DSCAgent.load(th.data_dir)

  assert isinstance(train_set, DSCSet)
  assert isinstance(val_set, DSCSet)
  assert isinstance(test_set, DSCSet)

  # input_shape and output_dim should be determined here
  if th.use_rnn: th.input_shape = train_set.features[0][0].shape
  else: th.input_shape = train_set.features[0].shape
  th.output_dim = train_set.num_classes

  train_set.report()
  val_set.report()
  test_set.report()

  return train_set, val_set, test_set



if __name__ == '__main__':
  from dsc_core import th

  th.data_config = 'rml:10-;iq'
  th.report_detail = True
  train_set, val_set, test_set = load()
  # print(train_set.SNRs)
  train_set.report()
  val_set.report()
  test_set.report()
