import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        downsampler = lambda n: nn.Sequential(
            nn.Conv2d(n, 2 * n, kernel_size=1, stride=(2, 2), bias=False),
            nn.BatchNorm2d(2 * n),
        )

        basic_block = lambda n: models.resnet.BasicBlock(n, 2 * n, 2, downsample=downsampler(n))

        model_ft = models.resnet18(pretrained=True)
        model_ft.layer4 = basic_block(256)
        model_ft.fc = nn.Linear(model_ft.fc.in_features, 512)

        self.backbone = model_ft

    def forward(self, x):
        x = self.backbone(x)
        x = F.normalize(x)
        return x


class VeRiNet(nn.Module):
    def __init__(self, checkpoint):
        super(VeRiNet, self).__init__()
        backbone = torch.load(checkpoint)
        backbone.backbone.fc = nn.Sequential(
            nn.Linear(backbone.backbone.fc.in_features, 512),
            # nn.ReLU(),
            # nn.BatchNorm1d(1024),
            # nn.Linear(1024, 512),
        )
        self.backbone = backbone

    def forward(self, x):
        x = self.backbone(x)
        x = F.normalize(x)
        return x
