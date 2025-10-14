import torch
import os
from skimage import img_as_ubyte
from utils.imgdata import getevalloader
import utils
import argparse
import math
from collections import OrderedDict
from tqdm import tqdm
from net_arch.model import CSTNet


parser = argparse.ArgumentParser()
parser.add_argument('--preprocess', type=str, default='crop')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. -1 for CPU')
parser.add_argument('--data_path', type=str, default='./dataset/test/input', help='test dataset`s input path')
parser.add_argument('--save_path', type=str, default='./results/test', help='test dataset`s output path')
parser.add_argument('--weights', type=str, default='./logs/net_best.pth')
parser.add_argument('--eval_workers', type=int, default=1)
opt = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids


def splitimage(imgtensor, crop_size=128, overlap_size=10):
    _, C, H, W = imgtensor.shape
    hstarts = [x for x in range(0, H, crop_size - overlap_size)]
    while hstarts[-1] + crop_size >= H:
        hstarts.pop()
    hstarts.append(H - crop_size)
    wstarts = [x for x in range(0, W, crop_size - overlap_size)]
    while wstarts[-1] + crop_size >= W:
        wstarts.pop()
    wstarts.append(W - crop_size)
    starts = []
    split_data = []
    for hs in hstarts:
        for ws in wstarts:
            cimgdata = imgtensor[:, :, hs:hs + crop_size, ws:ws + crop_size]
            starts.append((hs, ws))
            split_data.append(cimgdata)
    return split_data, starts


def get_scoremap(H, W, C, B=1, is_mean=True):
    center_h = H / 2
    center_w = W / 2

    score = torch.ones((B, C, H, W))
    if not is_mean:
        for h in range(H):
            for w in range(W):
                score[:, :, h, w] = 1.0 / (math.sqrt((h - center_h) ** 2 + (w - center_w) ** 2 + 1e-3))
    return score


def mergeimage(split_data, starts, crop_size=128, resolution=(1, 3, 128, 128)):
    B, C, H, W = resolution[0], resolution[1], resolution[2], resolution[3]
    tot_score = torch.zeros((B, C, H, W)).cuda()
    merge_img = torch.zeros((B, C, H, W)).cuda()
    scoremap = get_scoremap(crop_size, crop_size, C, B=B, is_mean=False)
    scoremap = scoremap.cuda()
    for simg, cstart in zip(split_data, starts):
        hs, ws = cstart
        merge_img[:, :, hs:hs + crop_size, ws:ws + crop_size] += scoremap * simg
        tot_score[:, :, hs:hs + crop_size, ws:ws + crop_size] += scoremap
    merge_img = merge_img / tot_score
    return merge_img

def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


if __name__ == '__main__':

    model_restoration = CSTNet.cuda()
    model_restoration.load_state_dict(torch.load(opt.weights))

    print("===>Testing using weights: ", opt.weights)
    model_restoration.cuda()
    model_restoration.eval()
    inp_dir = opt.data_path
    eval_loader = getevalloader(opt)
    result_dir = opt.save_path
    os.makedirs(result_dir, exist_ok=True)
    with torch.no_grad():
        for input_, file_ in tqdm(eval_loader):

            input_ = input_.cuda()
            B, C, H, W = input_.shape
            split_data, starts = splitimage(input_)
            for i, data in enumerate(split_data):
                split_data[i] = model_restoration(data)[1]
            restored = mergeimage(split_data, starts, resolution=(B, C, H, W))
            restored = restored.cpu()
            restored = torch.clamp(restored, 0, 1).permute(0, 2, 3, 1).numpy()

            for j in range(B):
                fname = file_[j]
                cleanname = fname
                save_file = os.path.join(result_dir, cleanname)
                utils.save_img(save_file, img_as_ubyte(restored[j]))
