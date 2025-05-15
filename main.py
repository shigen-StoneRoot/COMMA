import pickle
import matplotlib.pyplot as plt
import torch
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
from dataloading_new import init_training_dataloader, init_validation_dataloader, get_center
from losses import init_loss_func, init_DC_CE_loss_func
from medpy.metric import binary
from inference import init_predictor
import json
from models import COMMA
from skimage.transform import resize
from preprocessing import resample_data_or_seg_to_shape
from torch.optim.lr_scheduler import _LRScheduler


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


parser = argparse.ArgumentParser()
parser.add_argument('--ckdir', type=str, default="./pretraining_checkpoints")
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

parser.add_argument('--seed', type=int, default=1) # KiPA: 1 warm-up schedule
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--test_batch_size', type=int, default=1)

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
        train_subs = splits[0]['train']
        valid_subs = splits[0]['val']


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
                         glb_size=(96, 192, 192)):
    model.eval()
    model.set_deep_supervision(False)
    predictor = init_predictor(model, patch_size, inference_allowed_mirroring_axes, device)
    val_loss = 0.0


    with torch.no_grad():
        test_idxs = val_loader.get_test_idx()
        for idx in test_idxs:
            print('start to predict {}'.format(idx))
            img, gt, cur_spacing, target_spacing, ori_shape = val_loader.load_case(idx)
            # target_spacing = np.array([target_spacing[2], target_spacing[1], target_spacing[0]])

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

            logits = resample_data_or_seg_to_shape(logits, ori_shape, cur_spacing, target_spacing,
                                                   is_seg=False, order=1)
            assert np.array_equal(ori_shape, np.array(logits.shape[1:]))
            assert np.array_equal(ori_shape, np.array(gt.shape[1:]))

            logits = torch.from_numpy(logits).float()
            logits = softmax_helper_dim0(logits)
            segmentation = logits.argmax(0)

            seg = segmentation

            val_loss += compute_dice(seg.cpu().numpy().squeeze(), gt.squeeze())
        val_loss /= test_idxs.shape[0]

    model.set_deep_supervision(True)
    return val_loss


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
    lr_scheduler = PolyLRScheduler(optimizer, args.lr, args.total_epochs // 250)
    lamba = args.lamba
    init_it = 1

    for it in tqdm.tqdm(range(init_it, args.total_epochs + 1)):

        if it % 250 == 0:
            lr_scheduler.step(it // 250)

        model.train()
        optimizer.zero_grad()
        it_data = next(train_loader)
        x_loc, seg = it_data['data'].to(device), it_data['target']
        x_glb = it_data['glb_data'].to(device)
        y_glb = it_data['glb_seg'].to(device)


        center = it_data['center']
        flips = it_data['flips']
        angles = it_data['angles']

        glb_coord = it_data['glb_coord']

        model.set_flips_angles(flips, angles)
        model.set_glb_coord(glb_coord)
        seg = [s.to(device) for s in seg]

        if scaler is not None:
            with autocast():
                out, glb_out = model(x_glb, x_loc, center)
                loss = loss_func(out, seg)
                loss_glb = glb_loss_func(glb_out, y_glb)

                loss = loss + lamba * loss_glb
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
                scaler.step(optimizer)
                scaler.update()

        else:
            out = model(x_glb, x_loc, center)
            loss = loss_func(out, seg)
            loss.backward()
            optimizer.step()

    state_dict = model.state_dict()
    save_dict = {"state_dict": state_dict}
    torch.save(save_dict, os.path.join(r'./CKs/', args.dataset + '_COMMA_final.pt'))

if __name__ == '__main__':
    run(args)

