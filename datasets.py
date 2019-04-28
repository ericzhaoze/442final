import glob
import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms



cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Reading data from 2 folders seperately, ordered by name
class DatasetReader(Dataset):
    def __init__(self, root, mode="train"):
        

        self.filesA = sorted(glob.glob(os.path.join(root+"/A/", mode) + "/*.*"))
        if mode == "train":
            self.filesA.extend(sorted(glob.glob(os.path.join(root+"/A/", "test") + "/*.*")))

        self.filesB = sorted(glob.glob(os.path.join(root+"/B/", mode) + "/*.*"))
        if mode == "train":
            self.filesB.extend(sorted(glob.glob(os.path.join(root+"/B/", "test") + "/*.*")))

    def __getitem__(self, index):

        tensorfilter = transforms.Compose([transforms.ToTensor()])

        image_A = Image.open(self.filesA[index % len(self.filesA)]).convert('RGB')
        image_B = Image.open(self.filesB[index % len(self.filesB)]).convert('RGB')
        
        image_A = tensorfilter(image_A)
        image_B = tensorfilter(image_B)

        return {"A": image_A, "B": image_B}

    def __len__(self):
        return len(self.filesA)
        
def load_data(opt):
    os.makedirs("results/%s" % opt.dataroot, exist_ok=True)
    os.makedirs("checkpoints/%s" % opt.dataroot, exist_ok=True)

    train_dataloader = DataLoader(
        DatasetReader("./%s" % opt.dataroot),
        batch_size=opt.batch_size,
        shuffle=True,
    )

    val_dataloader = DataLoader(
        DatasetReader("./%s" % opt.dataroot, mode="val"),
        batch_size=1,
        shuffle=True,
    )
    return train_dataloader, val_dataloader
