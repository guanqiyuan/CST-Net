import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler
import utils
import argparse
from DerainDataset import TrainValDataset
from losses import *
from model_utils.RGB_YCbCr_tools import rgb_to_ycbcr
from net_arch.model import CSTNet


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./dataset', help='data set path')
parser.add_argument('--save_path', type=str, default='logs/', help='weights save path')
parser.add_argument('--epochs', type=int, default=501, help='num of train epoch')
parser.add_argument('--patch_size', type=int, default=128, help='training patch size')
parser.add_argument('--train_batch_size', type=int, default=4, help='training batch size')
parser.add_argument('--val_batch_size', type=int, default=4, help='val batch size')
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
parser.add_argument('--end_lr', type=float, default=1e-6, help='end learning rate')
parser.add_argument('--eval_freq', type=int, default=1)
parser.add_argument('--writer_freq', type=int, default=10, help='Frequency of log writing')
parser.add_argument('--warmup_epochs', type=int, default=5, help='warmup epochs')
opt = parser.parse_args()


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))
print('Loading dataset ...\n')

dataset_train = TrainValDataset("train", opt.data_dir, opt.patch_size)
train_loader = DataLoader(dataset=dataset_train, num_workers=0, batch_size=opt.train_batch_size, shuffle=True, drop_last=True)
print("# of training samples: %d\n" % int(len(dataset_train)))
dataset_val = TrainValDataset("validation", opt.data_dir, opt.patch_size)
val_loader = DataLoader(dataset=dataset_val, num_workers=0, batch_size=opt.val_batch_size, shuffle=True, drop_last=True)
print("# of validation samples: %d\n" % int(len(dataset_val)))

model = CSTNet.cuda()

writer = SummaryWriter(opt.save_path)
criterion_ssim = utils.SSIM().cuda()
criterion_mse = nn.MSELoss().cuda()
criterion_L1 = nn.L1Loss().cuda()
criterion_char = CharbonnierLoss()
criterion_edge = EdgeLoss()
criterion_psnr = PSNRLoss()

optimizer = torch.optim.Adam(model.parameters(),
                            lr=opt.lr,           
                            betas=(0.9, 0.999),
                            eps=1e-8)

scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs - opt.warmup_epochs, eta_min=opt.end_lr)
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=opt.warmup_epochs, after_scheduler=scheduler_cosine)

best_psnr = 0

for epoch in range(1, opt.epochs):
    epoch_start_time = time.time()
    for param_group in optimizer.param_groups:
        print('learning rate %f' % param_group["lr"])

    for i, (images, labels) in enumerate(tqdm(train_loader), 0):
        images = images.cuda()
        labels = labels.cuda()

        model.train()

        train_psnr_val_rgb = []
        model.zero_grad()
        optimizer.zero_grad()

        YCbCr_labels = rgb_to_ycbcr(labels)
        Y_labels = YCbCr_labels[:, 0, :, :]
        Y_labels = Y_labels.unsqueeze(1)
        Y_labels = Y_labels.cuda()

        Y_hat, out_image = model(images)
        Y_hat = Y_hat.cuda()
        out_image = out_image.cuda()

        mse = criterion_mse(out_image, labels)
        ssim = criterion_ssim(out_image, labels)
        loss_char = criterion_char(out_image, labels)
        loss_edge = criterion_edge(out_image, labels)
        loss_psnr = criterion_psnr(out_image, labels)
        loss_Y = criterion_mse(Y_hat, Y_labels)
        loss = loss_Y + loss_psnr + ssim + loss_char + loss_edge*0.5
        loss.backward()
        optimizer.step()

        for res, tar in zip(out_image, labels):
            train_psnr_val_rgb.append(utils.torchPSNR(res, tar))
        psnr_train = torch.stack(train_psnr_val_rgb).mean().item()

    if epoch % opt.writer_freq == 0:
        writer.add_scalar('loss', loss.item(), epoch)
        writer.add_scalar('ssim on training data', ssim, epoch)
        writer.add_scalar('mse on training data', mse, epoch)
        writer.add_scalar('PSNR on training data', psnr_train, epoch)
        writer.add_scalar('lr on training data', optimizer.param_groups[0]["lr"], epoch)

    if epoch % opt.eval_freq == 0:
        model.eval()
        psnr_val_rgb = []
        for ii, (images_1, labels_1) in enumerate(tqdm(val_loader), 0):
            input = images_1.cuda()
            target = labels_1.cuda()
            with torch.no_grad():
                restored = model(input)[1]
            for res, tar in zip(restored, target):
                psnr_val_rgb.append(utils.torchPSNR(res, tar))
        psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
        if psnr_val_rgb > best_psnr:
            best_psnr = psnr_val_rgb
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_best.pth'))
            torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_step%d.pth' % (epoch)))

    print("------------------------------------------------------------------")
    print(
        "Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tTrain_Best_PSNR: {:.4f}\tSSIM: {:.4f}\tMSE:{:.4f}\tLearningRate {:.8f}\tTest_Best_PSNR: {:.4f}".format(
            epoch, time.time() - epoch_start_time, loss.item(), psnr_train, ssim, mse, optimizer.param_groups[0]["lr"],
            best_psnr))
    print("------------------------------------------------------------------")

    scheduler.step()
