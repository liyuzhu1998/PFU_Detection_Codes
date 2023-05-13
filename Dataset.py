from torch.utils import data
import random
import os
from os import listdir
from os.path import join
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import scipy.ndimage as ndimage

def is_target_file(filename):
    return filename.endswith(".npy")


def load_img(filepath):
    y = np.load(filepath).astype(np.float32)
    return y


class DataFolder(data.Dataset):

    def __init__(self,in_channels,in_frames ,image_dir, lab, start_frame = 0, input_transform=None, tf_crop=False):
        super(DataFolder,self).__init__()

        self.in_channels = in_channels
        self.in_frames = in_frames
        self.start_frame = start_frame
        cur_label = np.asarray(lab)
        
        self.image_filenames = image_dir
        self.label = [cur_label]*len(self.image_filenames) # expand one item to #image_filenames items
    
    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self,index):
        path = self.image_filenames[index]
        s = self.start_frame
        input_raw = load_img(path)[:,s:self.in_frames+s,:,:].astype(float)
        label = self.label[index]

        input_raw = np.nan_to_num(input_raw)
        input = (input_raw + 3.14) / 6.28
        
        if random.randint(0,1):
            input = np.flip(input,axis=3).copy()
        if random.randint(0,1):
            input = np.flip(input,axis=2).copy()
        rot = random.randint(0,3)
        img = np.rot90(input, k=rot, axes=(2,3)).copy()

        return img,label,path

def get_data_set(in_channels,in_frames,image_dir,lab,start_frame = 0, input_transform=None, tf_crop=False):
    return DataFolder(in_channels,in_frames,image_dir,lab,start_frame,input_transform,tf_crop)