from args import *
from UNet_monai import Unet
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from dataset import SpaceNet7DataModule
import torchvision.transforms.functional as TF
from torchvision import transforms
import skimage.io as io
from torchvision.transforms import CenterCrop
import torch
import numpy as np

path = r"C:\Test\Github\SpaceNet7-Buildings-Detection\trained_models\best_model.ckpt"
pathImg = r"C:\Test\Building_data\images\0.png"
args = get_main_args()
model = Unet.load_from_checkpoint(checkpoint_path= path, args = args)
model.eval()
# model.load_state_dict(path)
# x = io.imread(pathImg)
# print(x.shape)

# dm = SpaceNet7DataModule(args = args)
# dm.setup()

# img = dm.spaceNet7_val
# print(np.array(img).shape)

# io.imshow(x)
# x = x.transpose((2,0,1))
# x = TF.to_tensor(x)

# i, j, h, w = transforms.RandomCrop.get_params(x, output_size=(1088, 1088))
# x = TF.crop(x, i, j, h, w)
# x = CenterCrop([512, 512])(x)
# x = torch.tensor(x, dtype=torch.float32)
# x = TF.to_tensor(x)
# normalize = transforms.Normalize(mean=[0.20702111, 0.26308049, 0.24522567],
#                                          std=[0.11984661, 0.11106335, 0.09017803]) 
# x = normalize(x)

# model(x)
# io.imsave("C:\Test\Github\SpaceNet7-Buildings-Detection\output_samples", model(x))
