import torch
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

# original image: 5000x5000
# cut into 100 input images: 500x500x3
# masks: 500x500
# crop into random part of size: 384 x 384 x c

# arg to change:
# base_dir: for data folder
# ckpt_path: for checkpoint path
# exec_mode: for train/evaluate
# img_size: input image size


def get_main_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    arg = parser.add_argument
    arg ("--seed", type=int, default=26012022, help="Random Seed")
    arg("--generator", default=torch.Generator().manual_seed(26012022), help='Train Validate Predict Seed')

    # arg("--base_dir", type=str, default="LiveEO_ML_intern_challenge", help="Data Directory")
    # data directory?
    arg("--base_dir", type=str, default=r"C:\Test\Building_data", help="Data Directory")

    arg("--img_size", type=int, default=500, help="input image size")

    arg ("--num_workers", type=int, default=2, help="DataLoader num_workers")
    arg ("--in_channels", type=int, default=4, help="#Input Channels")
    arg ("--out_channels", type=int, default=8, help="#Output Channels")
    arg ("--learning_rate", type=float, default=1e-4, help="Learning Rate")
    arg ("--weigh_decay", type=float, default=1e-5, help="Weigh Decay")
    arg ("--kernels",  default=[[3, 3]] * 5, help="Convolution Kernels")
    arg ("--strides",  default=[[1, 1]] +  [[2, 2]] * 4, help="Convolution Strides")

    arg("--num_epochs", type=int, default=20, help="Number of Epochs")
    arg("--exec_mode", type=str, default='train', help='Execution Mode')
    arg("--ckpt_path", type=str, default= r"C:\Test\Github\SpaceNet7-Buildings-Detection\trained_models\best_model.ckpt", help='Checkpoint Path')
    arg("--save_path", type=str, default='./', help='Saves Path')

    arg("--samples_per_epoch", type=int, default=1000, help="Random Samples Per Epoch")
    
    # arg("--crop_size", type=int, default=480, help="centered crop size")
    arg("--crop_size", type=int, default= 384, help="centered crop size") # must be multiple of 64

    arg("--batch_size", type=int, default=12, help="batch size")
    # return parser.parse_args(args=[])
    return parser.parse_args()




