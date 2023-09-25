from torch import nn
from torchvision.transforms import ToTensor
import torch

class LFIClassification(nn.Module):
    def __init__(self):
        super(LFIClassification, self).__init__()
        self.linear_stack = nn.Sequential(
            nn.Flatten(),
            nn.Unflatten(1, torch.Size([3, 8*8, 376, 541])),
            nn.Conv3d(in_channels=3, out_channels=16, kernel_size=(1,5,5), padding=(0,2,2), stride=(1,2,2)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Unflatten(1, torch.Size([16, 8, 8, 188*271])),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(5,5,1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Unflatten(1, torch.Size([32, 4*4, 188, 271])),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(1,5,5), padding=(0,2,2), stride=(1,2,2)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Unflatten(1, torch.Size([32, 4, 4, 94*136])),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(4,4,1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=32*94*136, out_features=500),
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=4),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        output = self.linear_stack(x)
        return output