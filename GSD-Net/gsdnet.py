import pickle

import torch.nn.init as init
from model.resnet1 import *


class GSDNet(nn.Module):
    def __init__(self, num_class=7):
        super(GSDNet, self).__init__()

        self.resnet = ResNet(Bottleneck, [3, 4, 6, 3],num_classes=num_class)
        with open('/data/vgg_msceleb_resnet50_ft_weight.pkl', 'rb') as f:
            obj = f.read()
        state_dict = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}
        self.resnet.load_state_dict(state_dict, strict=False)


    def forward(self, x):
        return self.resnet(x)



