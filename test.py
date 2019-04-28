from datasets import load_data
from torch.autograd import Variable
import torch
from torchvision.utils import save_image
from options import init_option
from network import *
from datasets import *
from pix2pix import *

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# same test results
def test(model, val_dataloader, opt):
    for i, batch in enumerate(val_dataloader):
        real_A = Variable(batch["B"].type(Tensor))
        real_B = Variable(batch["A"].type(Tensor))
        fake_B = model.generator(real_A)
        save_image(real_A.data, "results/%s/%s_real_A.png" % (opt.dataroot, i))
        save_image(fake_B.data, "results/%s/%s_fake_B.png" % (opt.dataroot, i))
        save_image(real_B.data, "results/%s/%s_real_B.png" % (opt.dataroot, i))

if __name__ == '__main__':
    opt = init_option().initialize()
    train_dataloader, val_dataloader = load_data(opt)
    
    model = pix2pix(opt, mode='test')

    test(model, val_dataloader, opt)

