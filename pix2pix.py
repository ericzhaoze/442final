####################################
# EECS 442 Winter 2019 final project
# main GAN model
####################################

import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.utils import save_image
import numpy as np
import math
import sys
from torch.autograd import Variable

from network import *


cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

class pix2pix(nn.Module):
    def __init__(self, opt, mode = 'train'):
        super(pix2pix, self).__init__()
        self.opt = opt
        self.output_size = (1, 128 // 2 ** 3 - 1, 128 // 2 ** 3 - 1)

        self.generator = GeneratorUNet()
        self.discriminator = NLayerDilatedDiscriminator()
        self.criterion_GAN = torch.nn.BCEWithLogitsLoss()
        self.criterion_pixelwise = torch.nn.L1Loss()

        if cuda:
            self.generator = generator.cuda()
            self.discriminator = discriminator.cuda()
            self.criterion_GAN.cuda()
            self.criterion_pixelwise.cuda()
        # initialize mode
        if mode == 'train':
            self.generator.apply(weights_init_normal)
            self.discriminator.apply(weights_init_normal)
        # load pretrained model
        else:
            self.generator.load_state_dict(torch.load("checkpoints/%s/generator.pth" % (opt.dataroot)))
            self.discriminator.load_state_dict(torch.load("checkpoints/%s/discriminator.pth" % (opt.dataroot)))

        # initialize optimizer
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=opt.lr, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    # one training step for generator
    def step_G(self, real_A, real_B):
        # training label
        ones = Variable(Tensor(np.ones((real_A.size(0), *self.output_size))))

        self.optimizer_G.zero_grad()
        # generating fake results
        self.fake_B = self.generator(real_A)
        # GAN loss
        self.gan_loss = self.criterion_GAN(self.discriminator(self.fake_B, real_A), ones)
        # pixel loss
        self.pix2pix_loss = self.criterion_pixelwise(self.fake_B, real_B)
        # total generator loss
        self.loss_G = self.gan_loss + 100 * self.pix2pix_loss

        self.loss_G.backward()

        self.optimizer_G.step()

    # one training step for discriminator
    def step_D(self, real_A, real_B):
        # training labels
        ones = Variable(Tensor(np.ones((real_A.size(0), *self.output_size))))
        zeros = Variable(Tensor(np.zeros((real_A.size(0), *self.output_size))))

        self.optimizer_D.zero_grad()
        # loss of real images
        realimg_loss = self.criterion_GAN(self.discriminator(real_B, real_A), ones)
        # loss of fake images
        fakeimg_loss = self.criterion_GAN(self.discriminator(self.fake_B.detach(), real_A), zeros)
        # total discriminator loss
        self.loss_D = 0.5 * (realimg_loss + fakeimg_loss)

        self.loss_D.backward()
        self.optimizer_D.step()

    def print_status(self, num_epoch, num_iter, total_len):
        sys.stdout.write(
            "\repoch: %d/%d, iter: %d/%d, G_loss: %f, D_loss: %f, pixel_loss: %f, gan_loss: %f"
            % (
                num_epoch,
                self.opt.num_epochs,
                num_iter,
                total_len,
                self.loss_G.item(),
                self.loss_D.item(),
                self.pix2pix_loss.item(),
                self.gan_loss.item(),
            )
        )
    def save_status(self):

        print("\nSaving models for this epoch...")
        torch.save(self.generator.state_dict(), "checkpoints/%s/generator.pth" % (self.opt.dataroot))
        torch.save(self.discriminator.state_dict(), "checkpoints/%s/discriminator.pth" % (self.opt.dataroot))


