import torch.nn as nn
import torch.nn.functional as F
import torch
import functools

# Generator is modifid from https://github.com/eriklindernoren/PyTorch-GAN

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d8 = self.down8(d6)
        u1 = self.up1(d8, d6)
        u3 = self.up3(u1, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        return self.final(u7)

# Discriminator is implemented by ourselves

class NLayerDilatedDiscriminator(nn.Module):
    def __init__(self, input_nc = 3, ndf=64, norm_layer=nn.BatchNorm2d):
        super(NLayerDilatedDiscriminator, self).__init__()

        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        self.l = nn.Sequential(nn.Conv2d(input_nc*2, ndf * 2, kernel_size=kw, stride=2, padding=1), 
                                nn.LeakyReLU(0.2, True))

        self.relu1 = nn.Sequential(nn.Conv2d(ndf*2, ndf*4, kernel_size=kw, stride=2, padding=1, bias=use_bias), 
                    nn.modules.InstanceNorm2d(ndf*4), nn.LeakyReLU(0.2, True))

        self.relu2 = nn.Sequential(nn.Conv2d(ndf*4, ndf*8, kernel_size=kw, stride=2, padding=1, bias=use_bias), 
                    nn.modules.InstanceNorm2d(ndf*8), nn.LeakyReLU(0.2, True))

        self.relu3 = nn.Sequential(nn.Conv2d(ndf*8, ndf*8, kernel_size=3, stride=1, padding=1, bias=use_bias), 
                    nn.modules.InstanceNorm2d(ndf*8), nn.LeakyReLU(0.2, True))

        self.atrous = nn.Sequential(nn.Conv2d(ndf*8, ndf*8, kernel_size=3, padding=2, dilation=2, bias=use_bias),
                    nn.modules.InstanceNorm2d(ndf*8), nn.LeakyReLU(0.2, True))

        self.atrous2 = nn.Sequential(nn.Conv2d(ndf*8, ndf*8, kernel_size=3, padding=4, dilation=4, bias=use_bias),
                    nn.modules.InstanceNorm2d(ndf*8), nn.LeakyReLU(0.2, True))

        self.atrous3 = nn.Sequential(nn.Conv2d(ndf*8, ndf*8, kernel_size=3, padding=8, dilation=8, bias=use_bias),
                    nn.modules.InstanceNorm2d(ndf*8), nn.LeakyReLU(0.2, True))

        self.clean = nn.Sequential(nn.Conv2d(ndf*8, ndf*8, kernel_size=3, stride=1, padding=1, bias=use_bias),
                    nn.modules.InstanceNorm2d(ndf*8), nn.LeakyReLU(0.2, True))

        self.lsgan = nn.Sequential(nn.Conv2d(ndf*8, 1, kernel_size=kw, stride=1, padding=1, bias=False))

    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        res = self.l(img_input)
        res = self.relu1(res)
        res = self.relu2(res)
        relu3 = self.relu3(res)

        res = self.atrous(relu3)
        res = self.atrous2(res)
        res = self.atrous3(res)

        res = relu3 + res
        res = self.clean(res)
        res = self.lsgan(res)
        # print(res.shape)
        return res
