import torch
import numpy as np
import skimage.io as io
from utils import get_files
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
import torchvision.transforms.functional as TF
from torchvision.transforms import CenterCrop


class SpaceNet7(Dataset):
    def __init__(self, files, img_size, crop_size, exec_mode):
        
        self.files = files
        self.img_size = img_size
        self.crop_size = crop_size
        self.exec_mode = exec_mode

        # checking the first time
        self.flag = True

        # class indexing
        classes = [ 0,  1,  2,  4,  5,  7,  8, 11]
        indexs = np.array(range(0, len(classes)))
        self.dict_index = {}
        for a, b in zip(classes, indexs):
            self.dict_index[a] = b


    def OpenImage(self, idx, invert=True):
        # image = io.imread(self.files[idx]['image'])[:,:,0:3] #shape (H, W, 3)
        # if invert:
        #     image = image.transpose((2,0,1))                 #shape (3, H, W)
        # return (image / np.iinfo(image.dtype).max) #render the values between 0 and 1

        image = np.load(self.files[idx]['image'])
        return image
       
    def indexing(self, mask):
        id_mask = np.reshape(mask, -1)
        id_mask = [self.dict_index[x] for x in id_mask]
        id_mask = np.reshape(id_mask, newshape = mask.shape)
        return id_mask
        
    def OpenMask(self, idx):
        # mask = io.imread(self.files[idx]['mask'])
        # return np.where(mask==255, 1, 0) #change the values to 0 and 1
        # return np.where(mask > 0, 1, 0) #change the values to 0 and 255
        mask = np.load(self.files[idx]['mask'])
        id_mask = self.indexing(mask)

        return id_mask
    
    def __getitem__(self, idx):
        # read the images and masks as numpy arrays
        # x = self.OpenImage(idx, invert=True)
        

        x = self.OpenImage(idx, invert=False)/255
        y = self.OpenMask(idx)
        # padd the images to have a homogenous size (500, 500, C) 
        # also convert 0-255 image to 0-1 image
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        x = CenterCrop([self.img_size, self.img_size])(x)
        y = CenterCrop([self.img_size, self.img_size])(y)
    
        # if it is the training phase, create random (C, 430, 430) crops
        # if it is the evaluation phase, we will leave the orginal size (C, 1024, 1024)
        if self.exec_mode =='train':

            # x, y = self.crop(x[None], y[None], self.crop_size)
            # print(x.shape, y.shape)
            # Random crop

            i, j, h, w = transforms.RandomCrop.get_params(x, output_size=(self.crop_size, self.crop_size))
            x = TF.crop(x, i, j, h, w)
            y = TF.crop(y, i, j, h, w)

            # x, y = x[0], y[0]
        
        # numpy array --> torch tensor
        x = x.type(torch.float32)
        y = y.type(torch.uint8)
        
        # normalize the images (image- image.mean()/image.std())

        # ImageNet mean and std, may not be accurate ...
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5],
                                         std=[0.229, 0.224, 0.225, 0.5]) 
        
        # if self.flag:
        #     print("\n CHECKING INPUT FOR MODEL ... \n")
        #     print(f"input image shape = {x.shape} with min, max value of: {x.min()}, {x.max()}")
        #     print(f"input mask shape = {y.shape} with min, max value of: {y.min()}, {y.max()}")
        #     print(f"unique value for y: {y.unique()}")
        #     print("\n")
        #     self.flag = False

        return normalize(x), y
    
    
    def __len__(self):
        return len(self.files)


class SpaceNet7DataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args  = args
        
    def setup(self, stage=None):
        files = get_files(self.args.base_dir)
        train_files, test_files = train_test_split(files, test_size=0.1, random_state=self.args.seed)
        self.spaceNet7_train = SpaceNet7(train_files, self.args.img_size, self.args.crop_size, self.args.exec_mode)
        self.spaceNet7_val = SpaceNet7(test_files, self.args.img_size, self.args.crop_size, self.args.exec_mode)

        
    def train_dataloader(self):
        train_sampler = self.ImageSampler(len(self.spaceNet7_train), self.args.samples_per_epoch)
        train_bSampler = BatchSampler(train_sampler, batch_size=self.args.batch_size, drop_last=True)
        return DataLoader(self.spaceNet7_train, batch_sampler=train_bSampler, num_workers=self.args.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.spaceNet7_val, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False)
    
    def predict_dataloader(self):
        return DataLoader(self.spaceNet7_val, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False)


    class ImageSampler(Sampler):
        def __init__(self, num_images=300, num_samples=500):
            self.num_images = num_images
            self.num_samples = num_samples

        def generate_iteration_list(self):
            return np.random.randint(0, self.num_images, self.num_samples)

        def __iter__(self):
            return iter(self.generate_iteration_list())

        def __len__(self):
            return self.num_samples
