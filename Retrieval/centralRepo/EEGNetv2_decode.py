from braindecode.models import EEGNetv4
import torch.nn as nn
import torch


def stack_network(shape, eegnet):
    return nn.Sequential(
        nn.LayerNorm(shape),
        eegnet
    )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Decoder:
    def __init__(self):
        self.device = device
        self.shape = (59, 250)
        self.eegnet = EEGNetv4(
            in_chans=self.shape[0],
            n_classes=5,
            input_window_samples=self.shape[1],
            final_conv_length='auto',
            pool_mode='mean',
            F1=8,
            D=20,
            F2=160,
            kernel_length=64,
            third_kernel_size=(8, 4),
            drop_prob=0.25
        )
        self.eegnet = stack_network(self.shape, self.eegnet)  # add normlizing layerW
        self.eegnet.to(self.device)
        self.path_ckpts = '/home/ncclab/CS/Latested_Code/decode/model/EEGNet_model/1114_ckpt_50(1)_2cls.pt'
        self.eegnet.load_state_dict(torch.load(self.path_ckpts))
        self.eegnet.eval()

    def eeg_decode(self, data):
        data = torch.from_numpy(data).float().to(self.device)
        data = data.unsqueeze(0)
        prediction = self.eegnet(data)
        return prediction.cpu().detach().numpy()