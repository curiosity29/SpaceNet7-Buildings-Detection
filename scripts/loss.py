import torch.nn as nn
from monai.losses import DiceLoss, FocalLoss
# from monai.losses.dice import *
# from monai.networks import one_hot
from torch.nn import CrossEntropyLoss

from torchmetrics import Dice


class LossSpaceNet7(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        # self.dice = DiceLoss(sigmoid=True, batch=True)
        # self.ce = FocalLoss(gamma=2.0, to_onehot_y=False)

        self.dice = Dice(num_classes = self.n_class, ignore_index = 1)
        self.ce = CrossEntropyLoss(reduction = 'mean')
        
    def _loss(self, p, y):
        # one hot n_class
        # one_hot_y = one_hot(y[:, None, ...], num_classes = self.n_class)

        dice_loss = self.dice(p, y)
        categorical_loss = self.ce(p, y)

        return dice_loss + categorical_loss

        # return self.ce(p,y)
    
    def forward(self, p, y):
        return self._loss(p, y)