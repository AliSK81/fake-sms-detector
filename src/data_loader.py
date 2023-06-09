import numpy as np


class SMSDataLoader:
    def __init__(self):
        pass

    def load_data(self, data_path):
        with open(data_path) as file:
            sms_data_str = file.read()
            return self._process_data(sms_data_str)

    def _process_data(self, sms_data_str):
        data_arr = []

        data_records = sms_data_str.split('\n')[:-1]
        for data in data_records:
            label = None
            sample = None
            if data[:3] == 'ham':
                label = 0
                sample = data[4:]
            elif data[:4] == 'spam':
                label = 1
                sample = data[5:]
            else:
                label = 'N/A'

            data_arr.append([label, sample])

        data_arr = np.array(data_arr)
        data_label = data_arr[:, 0]
        data_records = data_arr[:, 1]

        return data_records, data_label
