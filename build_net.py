import torch.nn as nn
import torch.nn.functional as F
import torch


class ZeroShotNet(nn.Module):
    def __init__(self):
        # 执行父类的构造函数
        super(ZeroShotNet, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, (3, 3), padding=1),
            nn.Conv2d(32, 64, (3, 3), padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(), 
            nn.Conv2d(64, 128, (3, 3), padding=1),
            nn.ReLU(), 
            nn.Conv2d(128, 128, (3, 3), padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(), 
            nn.Conv2d(128, 256, (3, 3), padding=1),
            nn.ReLU(), 
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(), 
        )


    def forward(self, x):
        output =  self.layers(x)
        output = torch.flatten(output,1)
        fc1 = nn.Linear(output.shape[-1], 100)
        output = fc1(output)
        # fc2 = nn.Linear(output.shape[-1], 100)
        # output = fc2(output)
        return output

