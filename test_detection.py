# from __future__ import print_function
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
import getopt
import sys
from configobj import ConfigObj
from tqdm import tqdm
import os
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
import torchvision.utils as vutils
import torch.nn.functional as F
import pickle
# from Dataset import get_data_set
import network_v3 as network
import time
import shutil

import h5py as h5
from pathlib import Path
from tqdm import tqdm
import scipy.io


class DataFolder(data.Dataset):


    def __init__(self,in_channels,in_frames,raw_data_dir,lab,time_frame=16,timestep=1,stepsize=50,cropsize=500, input_transform=None):
        super(DataFolder,self).__init__()

        self.raw_data_dir = raw_data_dir
        self.label = lab
        self.in_channels = in_channels
        self.time_frame = time_frame
        self.stepsize = stepsize
        self.cropsize = cropsize
        self.phase_stack = []
        self.in_frames = in_frames

        time_start = time.time()
        for jj in range(time_frame-3*timestep,time_frame+1,timestep):
            phase_file = join(raw_data_dir,f"{jj}h",'_full_phase.mat')
            print(f"loading {phase_file}")
            try:
                mat_contents = scipy.io.loadmat(phase_file)
            except:
                mat_contents = h5.File(phase_file, 'r')
            full_phase = mat_contents['full_phase']
            self.phase_stack.append(full_phase)
        
        time_end = time.time()
        print('time loading data 1:', time_end-time_start, 's')
        time_start = time.time()
        
        self.phase_stack = np.asarray(self.phase_stack)

        time_end = time.time()
        print('time loading data 2:', time_end-time_start, 's')
        time_start = time.time()
        self.input_transform = input_transform

        self.phase_stack = np.transpose(self.phase_stack,(0,2,1))
        _,h1,h2 = self.phase_stack.shape
        time_end = time.time()
        print('time loading data 3:', time_end-time_start, 's')
        time_start = time.time()
        
        self.x1 = (h1-self.cropsize)//self.stepsize+1
        self.x2 = (h2-self.cropsize)//self.stepsize+1

        self.phase_stack = self.phase_stack[np.newaxis,...]
        time_end = time.time()
        print('time loading data 4:', time_end-time_start, 's')
        time_start = time.time()

        self.phase_stack[0,:,:,:] = (self.phase_stack[0,:,:,:]+3.14)/6.28

        time_end = time.time()
        print('time loading data 5:', time_end-time_start, 's')

    
    def __len__(self):
        return self.x1*self.x2

    def __getitem__(self,index):
        p1 = index//self.x2
        p2 = index%self.x2

        x1 = self.stepsize*p1
        x2 = self.stepsize*p2

        input = self.phase_stack[:,0:self.in_frames,x1:x1+self.cropsize,x2:x2+self.cropsize]
        

        label = self.label
        if self.input_transform:
            input = input.copy()
            input = self.input_transform(input)

        return input,label,index+1

def get_data_set(in_channels,in_frames,raw_data_dir,lab,time_frame=16,timestep=1,stepsize=50,cropsize=500,input_transform=None):
    return DataFolder(in_channels,in_frames,raw_data_dir,lab,time_frame,timestep,stepsize,cropsize,input_transform)


if __name__ == '__main__':
    for timepoint in range(15,16,1): #(strat time: end time: timestep)
        modelPath = './model_detection_epoch264.pth' #60
        inputfolder = "./Network input/Negative"

        log_folder = Path(os.path.join(inputfolder,'log')).mkdir(parents=True,exist_ok=True)
        result_file = os.path.join(inputfolder,'log','result_480pixels_epoch264_high_speed_negative'+str(timepoint)+'.txt')
        result_file = open(result_file,'w')
        
        input_png_file = inputfolder.replace('npy','tiff')

        print('loading model')
        model = network.DenseNet(in_channels=1,in_frames=4,init_channels=64,growth_rate=8,blocks = [3,3,6,9],num_classes=2,drop_rate=0.5, bn_size = 16, batch_norm = True)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        model.to(device)
        model.load_state_dict(torch.load(modelPath))
        model.eval()
        
        time_start = time.time()
        dataset = get_data_set(in_channels = 1, in_frames = 4, raw_data_dir = inputfolder, lab = 0, time_frame = timepoint, timestep = 1, stepsize = 64, cropsize = 480) # 64 480
        data_loader = DataLoader(dataset=dataset, num_workers=3,
                            batch_size=100, shuffle=False, drop_last=False,
                            pin_memory=True)
        time_end = time.time()
        print('time loading data', time_end-time_start, 's')

        time_start = time.time()
        first_line = True
        with torch.no_grad():
            for i,batch in enumerate(tqdm(data_loader)):
                input = batch[0].float().to(device)
                indices = batch[2]
                output,_ = model(input)
                _,predicted = torch.max(output,1)
                for j,lab in enumerate(predicted):
                    if first_line:
                        result_file.write(f'{indices[j]} {output[j,0]} {output[j,1]}')
                        first_line = False
                    else:
                        result_file.write(f'\n{indices[j]} {output[j,0]} {output[j,1]}')

        result_file.close()
        time_end = time.time()
        print('time cost', time_end-time_start, 's')
