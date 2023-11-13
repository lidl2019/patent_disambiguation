import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps
import torch.nn as nn
from torch import optim
import torch.nn.functional as F


class PatentNetwork(nn.Module):
    def __init__(self, input_size = 768):
        super(PatentNetwork, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),

            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),

            # nn.Linear(1024, 1024),
            # nn.ReLU(),
            # nn.BatchNorm1d(1024),
            #
            # nn.Linear(1024, 1024),
            # nn.ReLU(),
            # nn.BatchNorm1d(1024),

            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(256, 32),
            nn.Sigmoid()
        )

    def set_dimension(self, size):
        self.input_size = size

    def forward(self, x):
        output = self.fc1(x)
        output = self.fc2(output)
        return output

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    # label==1: Same Inventor
    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive