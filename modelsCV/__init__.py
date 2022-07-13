import torch
from torchvision import models
import torch.nn as nn


class MyresNet18(torch.nn.Module):
    def __init__(self):
        super(MyresNet18, self).__init__()
        model = models.resnet18(pretrained = True)
        self.resnet_lay=nn.Sequential(*list(model.children())[:-2])
        self.conv1_lay = nn.Conv2d(512, 256, kernel_size = (1,1),stride=(1,1))
        self.relu1_lay = nn.ReLU(inplace = True)
        self.drop_lay = nn.Dropout2d(0.5)
        self.global_average = nn.AdaptiveAvgPool2d((1,1))
        self.fc_Linear_lay2 = nn.Linear(256,2)
        

    def load(self, path):
       
       self.load_state_dict(torch.load(path,map_location=torch.device('cpu')))

    def forward(self, x):
        x= self.resnet_lay(x)
        x = self.conv1_lay(x)
        x = self.relu1_lay(x)
        x = self.drop_lay(x)
        x= self.global_average(x)
        x = x.view(x.size(0),-1)
        x = self.fc_Linear_lay2 (x)
        return x
