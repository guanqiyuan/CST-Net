import os
import numpy as np
from numpy.random import RandomState
from torch.utils.data import Dataset
from skimage import io, transform
import torch

class TrainValDataset(Dataset):
    def __init__(self, name, data_dir, patch_size):
        super(TrainValDataset, self).__init__()
        self.rand_state = RandomState(66)
        self.name = name
        self.root_dir = os.path.join(data_dir, self.name)         
        self.root_dir_rain = os.path.join(self.root_dir, "input")    
        self.root_dir_label = os.path.join(self.root_dir, "target")

        self.mat_files_rain = sorted(os.listdir(self.root_dir_rain))
        self.mat_files_label = sorted(os.listdir(self.root_dir_label))

        self.patch_size = patch_size

        self.file_num = len(self.mat_files_label)

    def __len__(self):
        if self.name == "train":
            return self.file_num * 1
        else:
            return self.file_num
        
    def __getitem__(self, idx):

        file_name_rain = self.mat_files_rain[idx % self.file_num] 
        file_name_label = self.mat_files_label[idx % self.file_num]

        img_file_rain = os.path.join(self.root_dir_rain, file_name_rain)
        img_file_label = os.path.join(self.root_dir_label, file_name_label)

        img_rain = io.imread(img_file_rain).astype(np.float32) / 255
        img_label = io.imread(img_file_label).astype(np.float32) / 255

        O, B = self.crop(img_rain, img_label)
        O = np.transpose(O, (2, 0, 1))
        B = np.transpose(B, (2, 0, 1))
        return torch.Tensor(O), torch.Tensor(B)

    def crop(self, img_rain, img_label):
        patch_size = self.patch_size

        h, w, c = img_rain.shape
        h = h - 1
        w = w - 1

        p_h, p_w = patch_size, patch_size

        r = self.rand_state.randint(0, h - p_h)
        c = self.rand_state.randint(0, w - p_w)

        O = img_rain[r: r + p_h, c: c + p_w]
        B = img_label[r: r + p_h, c: c + p_w]
        return O, B
