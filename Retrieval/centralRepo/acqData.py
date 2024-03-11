import numpy as np
from Jellyfish_Python_API.neuracle_api import DataServerThread
import mne
import time


class AcqData:
    def __init__(self, downsample = 125, t_buffer = 3):
        
        self.sample_rate = 1000
        self.t_buffer = t_buffer
        self.downsample = downsample
        self.thread_data_server = DataServerThread(self.sample_rate, t_buffer)
        self.start_server()

    def start_server(self):
        
        notconnect = self.thread_data_server.connect(hostname='192.168.3.37', port=8712)
        # notconnect = self.thread_data_server.connect(hostname='127.0.0.1', port=8712)
        if notconnect:
            raise TypeError("Can't connect JellyFish, Please open the hostport ")
        else:
            
            while not self.thread_data_server.isReady():
                time.sleep(1)
                continue
            
            self.thread_data_server.start()
            print('Data server start')

    def get_current_data(self):
        all_data = self.thread_data_server.GetBufferData()
        data = all_data[:59, :]
        triggers = all_data[-1]
        return data, triggers

    def npy2raw(self, npy_data):
        
        n_channels, n_times = npy_data.shape
        
        sfreq = 1000  
        ch_names = [
            'Fpz', 'Fp1', 'Fp2', 'AF3', 'AF4', 'AF7', 'AF8', 'Fz', 'F1', 'F2', 'F3', 'F4',
            'F5', 'F6', 'F7', 'F8', 'FCz', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FT7',
            'FT8', 'Cz', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'T7', 'T8', 'CP1', 'CP2', 'CP3',
            'CP4', 'CP5', 'CP6', 'TP7', 'TP8', 'Pz', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'POz',
            'PO3', 'PO4', 'PO5', 'PO6', 'PO7', 'PO8', 'Oz', 'O1', 'O2'
        ]  # , 'ECG', 'HEOR', 'HEOL', 'VEOU', 'VEOL'
        ch_types = ['eeg'] * n_channels
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types, verbose=False)

        raw = mne.io.RawArray(npy_data, info, verbose=False)
        return raw

    def preprocessing(self, npy_data):
        raw_data = self.npy2raw(npy_data)
        raw_data.notch_filter(50, verbose='ERROR')
        raw_data.filter(1, 40, verbose='ERROR')
        raw_data.resample(self.downsample)
        return raw_data.get_data()

    def single_process(self, data):
        data = data.reshape(1, len(data))
        print(data.shape)
        ch_name = ['Show']
        ch_type = ['eeg']
        sfreq = 1000
        info = mne.create_info(ch_names=ch_name, sfreq=sfreq, ch_types=ch_type)
        raw = mne.io.RawArray(data, info, verbose=False)
        raw.notch_filter(50, verbose=False)
        raw.filter(1, 40, verbose=False)
        raw.resample(self.downsample)
        raw_data = raw.get_data()
        return raw_data

    def stop_server(self):
        self.thread_data_server.stop()

if __name__ == '__main__':
    acq = AcqData()




