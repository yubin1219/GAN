import torch
import torch.nn as nn

class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d:
        super(UnetGenerator, self).__init__()
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)

# Defines the submodule with skip connection.
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc, submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d):
        super(UnetSkipConnectionBlock, self).__init__()
        if input_nc is None:
            input_nc = outer_nc
            
        self.outermost = outermost   
        
        if outermost:
            self.down = nn.Sequential(nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias),
                                      submodule)
            self.up = nn.Sequential(nn.ReLU(True),
                                    nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1),
                                    nn.Tanh()
                                   )
            
        elif innermost:
            self.down = nn.Sequential(nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias))
            self.up = nn.Sequential(nn.ReLU(True),
                                    nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias),
                                    norm_layer(outer_nc)
                                   )
            
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            self.down = nn.Sequential(nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias),
                                      norm_layer(inner_nc),
                                      submodule)
            self.up = nn.Sequential(nn.ReLU(True),
                                    nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias),
                                    norm_layer(outer_nc)
                                   )

    def forward(self, x):
        if self.outermost:
            out = self.down(x)
            out = self.up(out)
            return out
        else:
            out = self.down(x)
            out = self.up(out)
            return torch.cat([x, out], 1)
