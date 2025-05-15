import pickle

import matplotlib.pyplot as plt
import torch

torch.multiprocessing.set_sharing_strategy('file_system')
import argparse
from monai import transforms
from monai.inferers import sliding_window_inference
from monai.metrics import RMSEMetric, MAEMetric, DiceMetric
import numpy as np
import random
from torch import optim, nn
from functools import partial
from baselines import PlainConvUNet, InitWeights_He
import os
import time
from torch.cuda.amp import autocast, GradScaler
from monai.losses import DiceCELoss, DiceFocalLoss
import nibabel as nib
import tqdm
import math
from einops import rearrange
from utils import load_dataloader
import torch.nn.functional as F
# from dataloading import init_training_dataloader, init_validation_dataloader
from dataloading_new import init_training_dataloader, init_validation_dataloader, get_center
from losses import init_loss_func, init_DC_CE_loss_func
from medpy.metric import binary
from inference import init_predictor
from torch.optim.lr_scheduler import _LRScheduler
import json
from models import COMMA
from skimage.transform import resize
from preprocessing import resample_data_or_seg_to_shape
import SimpleITK as sitk
from skimage.morphology import skeletonize, skeletonize_3d


def arr2nii_ITK(new_arr, ori_nii_path, new_nii_path):
    input_path = ori_nii_path
    output_path = new_nii_path

    image = sitk.ReadImage(input_path)

    new_array = new_arr.astype(float)

    new_image = sitk.GetImageFromArray(new_array)

    new_image.CopyInformation(image)

    sitk.WriteImage(new_image, output_path)


class PolyLRScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr


def softmax_helper_dim0(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, 0)


def compute_dice(seg, gt):
    return binary.dc(seg, gt)

def cl_score(v, s):
    return np.sum(v*s)/np.sum(s)


def compute_clDice(v_p, v_l):
    if len(v_p.shape)==2:
        tprec = cl_score(v_p,skeletonize(v_l))
        tsens = cl_score(v_l,skeletonize(v_p))
    elif len(v_p.shape)==3:
        tprec = cl_score(v_p,skeletonize_3d(v_l))
        tsens = cl_score(v_l,skeletonize_3d(v_p))
    return 2*tprec*tsens/(tprec+tsens)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="KiPA")
parser.add_argument('--device', type=str, default="0")
parser.add_argument('--lamba', type=float, default=0.25)
parser.add_argument('--image_size', type=tuple, default=(96, 96, 96))
parser.add_argument('--glb_size', type=tuple, default=(96, 256, 256))

parser.add_argument("--lr", default=1.5e-2, type=float, help="learning rate")
parser.add_argument('--weight_decay', type=float, default=3e-5,
                    help='weight decay (default: 0.05)')
parser.add_argument('--warmup_epochs', type=int, default=100)
parser.add_argument('--total_epochs', type=int, default=25000)
parser.add_argument('--reuse_ck', type=int, default=0)

parser.add_argument('--seed', type=int, default=111)  # 1
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--test_batch_size', type=int, default=1)

parser.add_argument("--RandFlipd_prob", default=0.5, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.5, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandRotate_prob", default=0.5, type=float, help="RandRotate aug probability")
parser.add_argument("--RandRotate_angle", default=[-0.35, 0.35], type=list, help="RandRotate angle range")
parser.add_argument("--GaussianNoise_prob", default=0.5, type=float, help="Gaussian Noise aug probability")
parser.add_argument("--GaussianSmooth_prob", default=0.5, type=float, help="Gaussian Smooth aug probability")
parser.add_argument("--GaussianSmooth_kernel", default=(0.65, 1.5), type=tuple, help="Gaussian Smooth kernel range")
parser.add_argument("--AdjustContrast_prob", default=0.5, type=float, help="Contrast aug probability")
parser.add_argument("--AdjustContrast_gamma", default=(0.25, 1.5), type=tuple, help="Adjust Contrast gamma range")

parser.add_argument("--RandScaleIntensityd_prob", default=0.5, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.5, type=float, help="RandShiftIntensityd aug probability")

args = parser.parse_args()


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = 0.0 + (args.lr - 0.0) * 0.5 * \
             (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.total_epochs - args.warmup_epochs)))
    for num, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr
    return lr


def init_dataloader(args):
    split_file = r'./Data/raw_data/{}/splits_final.json'.format(args.dataset)
    data_dir = r'./Data/preprocessed_data/{}'.format(args.dataset)

    with open(split_file, 'r') as f:
        splits = json.load(f)
        train_subs = splits[fold]['train']
        valid_subs = splits[fold]['val']

    train_loader = init_training_dataloader(data_dir, train_subs,
                                            patch_size=args.image_size,
                                            glb_size=args.glb_size, batch_size=args.batch_size)
    val_loader = init_validation_dataloader(data_dir, valid_subs)

    return train_loader, val_loader


def init_model_optimizer(args):
    model = COMMA()

    model.apply(InitWeights_He(1e-2))

    optimizer = optim.SGD(model.parameters(), args.lr, weight_decay=args.weight_decay,
                          momentum=0.99, nesterov=True)

    return model, optimizer


def validation_one_epoch(model, patch_size, inference_allowed_mirroring_axes, val_loader, device,
                         raw_img_path, sv_seg_path,
                         glb_size=(96, 256, 256)):
    model.eval()
    model.set_deep_supervision(False)
    predictor = init_predictor(model, patch_size, inference_allowed_mirroring_axes, device)
    val_dice, val_cldice = 0.0, 0.0
    subs = val_loader.subs

    with torch.no_grad():
        test_idxs = val_loader.get_test_idx()
        for idx in tqdm.tqdm(test_idxs):
            print('start to predict {}'.format(subs[idx]).replace('.npz', '.nii.gz'))
            img, gt, cur_spacing, target_spacing, unresample_shape, uncrop_shape, min_coords, max_coords = \
                val_loader.get_eval_case(idx)
            glb_data = np.zeros((1, 1, glb_size[0], glb_size[1], glb_size[2]),
                                dtype=np.float32)
            for c in range(glb_data.shape[0]):
                cur_glb = resize(img[c], glb_size, order=3)
                cur_glb = (cur_glb - np.mean(cur_glb)) / np.std(cur_glb)
                glb_data[:, c] = cur_glb

            img = torch.from_numpy(img).float()
            glb_data = torch.from_numpy(glb_data).float()

            img = img.to(device)
            glb_data = glb_data.to(device)
            predictor.set_glb_data(glb_data)
            predictor.set_glb_coord([np.array(img.shape[1:])])

            logits = predictor.predict_sliding_window_return_logits(img)

            logits = resample_data_or_seg_to_shape(logits, unresample_shape, cur_spacing, target_spacing,
                                                   is_seg=False, order=1)
            assert np.array_equal(unresample_shape, np.array(logits.shape[1:]))
            assert np.array_equal(unresample_shape, np.array(gt.shape[1:]))

            logits = torch.from_numpy(logits).float()
            logits = softmax_helper_dim0(logits)
            segmentation = logits.argmax(0)

            seg = segmentation

            seg = seg.cpu().numpy().squeeze()
            gt = gt.squeeze()

            if np.array_equal(uncrop_shape, np.array(seg.shape)) is not True:
                seg = val_loader.recover_ori_shape(seg, uncrop_shape, min_coords, max_coords)
                gt = val_loader.recover_ori_shape(gt, uncrop_shape, min_coords, max_coords)
            val_dice += compute_dice(seg, gt)
            val_cldice += compute_clDice(seg, gt)

            sub_nii = subs[idx].replace('.npz', '.nii.gz')
            ori_nii_path, new_nii_path = os.path.join(raw_img_path, sub_nii), os.path.join(sv_seg_path, sub_nii)
            arr2nii_ITK(seg, ori_nii_path, new_nii_path)

        val_dice /= test_idxs.shape[0]
        val_cldice /= test_idxs.shape[0]

        print('Dice: {}  ||  clDice: {}'.format(val_dice, val_cldice))


def run(args):
    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda:{}'.format(args.device))

    train_loader, val_loader = init_dataloader(args)

    model, optimizer = init_model_optimizer(args)

    weights = torch.tensor([1, 1 / 2, 1 / 4, 1 / 8, 1 / 16])
    weights = weights / weights.sum()
    loss_func = init_loss_func(weights).to(device)
    glb_loss_func = init_DC_CE_loss_func().to(device)

    model = model.to(device)


    scaler = GradScaler()

    lamba = args.lamba
    init_it = 1


    raw_img_path, sv_seg_path = r'./Data/raw_data/{}/masks'.format(args.dataset), \
                                r'./Prediction/{}'.format(args.dataset)
    if os.path.exists(sv_seg_path) is not True:
        os.mkdir(sv_seg_path)

    ck_file = os.path.join(r'./CKs/', args.dataset + '_COMMA_final.pt')
    checkpoint = torch.load(ck_file)
    model.load_state_dict(checkpoint['state_dict'])
    print('load checkpoint!')


    validation_one_epoch(model, args.image_size, (0, 1, 2), val_loader, device,
                         raw_img_path=raw_img_path, sv_seg_path=sv_seg_path,
                         glb_size=args.glb_size)





if __name__ == '__main__':
    run(args)

