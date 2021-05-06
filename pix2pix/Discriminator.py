import torch
import torch.nn as nn

class Discriminator(nn.Module):
  def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False) :
    super(Discriminator, self).__init__()
    self.conv1 = nn.Sequential(
      nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
      nn.LeakyReLU(0.2, True)
      )
    self.conv2 = nn.Sequential(
      nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
      norm_layer(ndf * 2),
      nn.LeakyReLU(0.2, True)
      )
    self.conv3 = nn.Sequential(
      nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
      norm_layer(ndf * 4),
      nn.LeakyReLU(0.2, True)
      )
    self.conv4 = nn.Sequential(
      nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
      norm_layer(ndf * 8),
      nn.LeakyReLU(0.2, True)
      )
    
    if use_sigmoid:
      self.conv5 = nn.Sequential(
        nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1),
        nn.Sigmoid()
        )
    else:
      self.conv5 = nn.Sequential(
        nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1)
        )
    
    def forward(self, x):
      x = self.conv1(x)
      x = self.conv2(x)
      x = self.conv3(x)
      x = self.conv4(x)
      x = self.conv5(x)
      return x
