import numpy as np

import torch
from centralRepo.networks import FBCNet

config = {}

config['modelArguments'] = {'nChan': 22, 'nTime': 1000, 'dropoutP': 0.5,
                                    'nBands': 9, 'm': 32, 'temporalLayer': 'LogVarLayer',
                                    'nClass': 4, 'doWeightNorm': True}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Decoder:
    def __init__(self):
        self.device = device
        self.netInitState = torch.load('E:/vscodeProject/imaginationControl/model/FBCNet_0.pth')
        self.model = FBCNet(**config['modelArguments'])
        self.model.load_state_dict(self.netInitState, strict=False)
        self.model.to(self.device)
        self.model.eval()

    def eeg_decode(self, data: np.ndarray) -> np.ndarray:
        print("eeg_decode")
        data = torch.from_numpy(data).float()
        print("data")
        data = data.unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(1, 1, 1, 1, 9).to(self.device)
        print("data")
        prediction = self.model(data)
        return prediction.cpu().detach().numpy()
