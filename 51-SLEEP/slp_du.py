import os, sys

from pictor import Pictor
from slp.sleep_record import SleepRecord
from slp.slp_set_loc import Monitor


def load_data():
  # Load data
  # ...
  # sleep_data_list = [SleepData(), SleepData(), ]
  path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'ucddb')
  sleep_data_list = SleepRecord.read_sleep_records(path)
  sleep_data_list_preprocessed = preprocess_data(sleep_data_list)


def preprocess_data(sleep_data_list):
  for sleep_data in sleep_data_list:
    eeg_data = sleep_data.data_dict['EEG']
    sleep_data.data_dict['EEG'][:, 1] = preprocess_data_each(eeg_data[:, 1])
    sleep_data.data_dict['EEG'][:, 2] = preprocess_data_each(eeg_data[:, 2])
    ecg_data = sleep_data.data_dict['ECG']
    sleep_data.data_dict['ECG'][:, 1] = preprocess_data_each(ecg_data[:, 1])
    sleep_data.data_dict['ECG'][:, 2] = preprocess_data_each(ecg_data[:, 2])
    sleep_data.data_dict['ECG'][:, 3] = preprocess_data_each(ecg_data[:, 3])
  print("*************************************")
  print("stage_ann:", sleep_data_list[0].data_dict['stage'][:10])
  print("*************************************")
  return sleep_data_list

def preprocess_data_each(data):
  import numpy as np
  from scipy import signal
  #滤波
  b, a = signal.butter(7, 0.64)
  filted_data = signal.filtfilt(b, a, data)
  #归一化
  arr_mean = np.mean(filted_data)
  arr_std = np.std(filted_data)
  precessed_data = (filted_data - arr_mean) / arr_std
  return precessed_data


if __name__ == '__main__':

  sleep_data_list = load_data()

  # Initiate a pictor
  p = Pictor(title='Sleep Monitor', figure_size=(15, 9))

  # Set plotter
  m = Monitor()
  p.add_plotter(m)

  # Set objects
  p.objects = sleep_data_list

  # Begin main loop
  p.show()
