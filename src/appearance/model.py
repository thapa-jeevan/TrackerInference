import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        model_ft = models.resnet18(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 128)

        self.backbone = model_ft

    def forward(self, x):
        x = self.backbone(x)
        x = F.normalize(x)
        return x
