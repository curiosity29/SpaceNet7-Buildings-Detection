from args import *
from UNet_monai import Unet
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from dataset import SpaceNet7DataModule
import torchvision.transforms.functional as TF
import skimage.io as io
from torchvision.transforms import CenterCrop

path = r"C:\Test\Github\SpaceNet7-Buildings-Detection\trained_models\best_model.ckpt"
pathImg = r"C:\Test\Building_data\images\0.png"
args = get_main_args()
model = Unet.load_from_checkpoint(checkpoint_path= path, args = args)
model.eval()
# model.load_state_dict(path)
x = io.imread(pathImg)
print(x.shape)
# io.imshow(x)
x = TF.to_tensor(x)
# x = CenterCrop([1088, 1088])(x)
# x = torch.tensor(x, dtype=torch.float32)
# x = TF.to_tensor(x)
# x = x.transpose((2,0,1))
# model(x)
io.imsave("C:\Test\Github\SpaceNet7-Buildings-Detection\output_samples", model(x))