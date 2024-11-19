import torch
import math
from torch import nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385

    def __init__(self, in_dim, out_dim, norm, factor_bottle=2, kernel=3, groups=1):
        super(Bottleneck, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        middle_dim = int(in_dim/factor_bottle)

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv1d(in_dim, middle_dim, kernel_size=1, padding=0, bias=False)
        self.bn1 = norm(middle_dim)
        self.conv2 = nn.Conv1d(middle_dim, middle_dim, kernel_size=kernel, padding=kernel//2, groups=groups, bias=False)
        self.bn2 = norm(middle_dim)
        self.conv3 = nn.Conv1d(middle_dim, out_dim, kernel_size=1, padding=0, bias=False)
        self.bn3 = norm(out_dim)
        self.relu = nn.ReLU(inplace=False)

        if in_dim != out_dim:
            self.downsample = nn.Sequential(
                                        nn.Conv1d(self.in_dim, self.out_dim, kernel_size=1, bias=False),
                                        norm(self.out_dim))
        else:
            self.downsample = None

    def _init_weights(self):

        torch.nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
#        self.conv1.bias.data.fill_(0.01)        
        torch.nn.init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
#        self.conv1.bias.data.fill_(0.01)
        torch.nn.init.kaiming_normal_(self.conv3.weight, mode='fan_in', nonlinearity='relu')
#        self.conv1.bias.data.fill_(0.01)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.in_dim != self.out_dim:
            identity = self.downsample(x)


        out = torch.add(out,  identity)
        out = F.relu(out)

        return out

