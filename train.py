from options import init_option

import os
import numpy as np
import math
import itertools

import sys

from network import *
from datasets import *
from pix2pix import *
from test import *

import torch.nn as nn
import torch.nn.functional as F
import torch


cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

if __name__ == '__main__':
    # get input parameters
    opt = init_option().initialize()
    # load data
    train_dataloader, val_dataloader = load_data(opt)

    model = pix2pix(opt)

    for epoch in range(opt.num_epochs):
        for i, batch in enumerate(train_dataloader):

            real_A = Variable(batch["B"].type(Tensor))
            real_B = Variable(batch["A"].type(Tensor))
            
            # train generator
            model.step_G(real_A, real_B)
            # train discriminator
            model.step_D(real_A, real_B)
            model.print_status(epoch, i, len(train_dataloader))

        model.save_status()
            