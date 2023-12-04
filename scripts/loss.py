import torch.nn as nn
from monai.losses import DiceLoss, FocalLoss
# from monai.losses.dice import *
# from monai.networks import one_hot
from torch.nn import CrossEntropyLoss
import torch

from torchmetrics import Dice


class LossSpaceNet7(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        # self.dice = DiceLoss(sigmoid=True, batch=True)
        # self.ce = FocalLoss(gamma=2.0, to_onehot_y=False)

        # self.dice = Dice(num_classes = self.n_class, ignore_index = 1)
        # self.dice = Dice(num_classes = self.n_class)

        # self.ce = CrossEntropyLoss(reduction = 'mean')
        self.ce = CrossEntropyLoss(ignore_index = 0)
        # self.ce = nn.BCEWithLogitsLoss()

    
        
    def _loss(self, p, y):
        # one hot n_class
        # one_hot_y = one_hot(y[:, None, ...], num_classes = self.n_class)

        # categorical_loss = self.ce(p, y)

        # return dice_loss + categorical_loss


        # dice_loss = self.dice(p, y)
        # return dice_loss
        return self.ce(p,y.long()) + self.dice_loss(p,y)


        # return self.ce(p,y)
    
    def forward(self, p, y):
        return self._loss(p, y)
    

    def dice_loss(logits, true, eps=1e-7):
        """Computes the Sørensen–Dice loss.

        Note that PyTorch optimizers minimize a loss. In this
        case, we would like to maximize the dice loss so we
        return the negated dice loss.

        Args:
            true: a tensor of shape [B, 1, H, W].
            logits: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output or logits of the model.
            eps: added to the denominator for numerical stability.

        Returns:
            dice_loss: the Sørensen–Dice loss.
        """
        num_classes = logits.shape[1]
        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
            pos_prob = torch.sigmoid(logits)
            neg_prob = 1 - pos_prob
            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:
            true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = F.softmax(logits, dim=1)
        true_1_hot = true_1_hot.type(logits.type())
        dims = (0,) + tuple(range(2, true.ndimension()))
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_loss = (2. * intersection / (cardinality + eps)).mean()
        return (1 - dice_loss)