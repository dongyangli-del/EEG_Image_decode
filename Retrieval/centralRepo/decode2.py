import numpy as np
from centralRepo import classifier
import torch

from centralRepo.networks import FBCNet

device = torch.device("cpu")
class Decoder:
    def __init__(self):
        self.device = device
        # self.autoencoder = torch.load('E:/vscodeProject/imaginationControl/model/best_autoencoder.pth')
        self.classifier = torch.load('/home/ncclab/CS/Latested_Code/decode/model/best_classifier.pth')

        # self.encoder = classifier.Autoencoder()
        self.result = classifier.Classifier()
        # self.encoder.load_state_dict(self.autoencoder, strict=False)
        self.result.load_state_dict(self.classifier, strict=False)

        # self.encoder.to(self.device)
        self.result.to(self.device)

        # self.encoder.eval()
        self.result.eval()

    def eeg_decode(self, data: np.ndarray) -> np.ndarray:
        data = torch.from_numpy(data).float()

        prediction = self.result(data)
        return prediction.cpu().detach().numpy()
