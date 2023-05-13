from __future__ import print_function
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
from torch.utils.data import Subset
from torch.utils.data.dataset import random_split
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
import math
import torch.nn.functional as F
import pickle
# import network as network
# import network_densenet_v2 as network
import network_v3 as network
from Dataset import get_data_set
from torch.nn import CrossEntropyLoss
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import math
import itertools
import random

def print_and_save_msg(msg, file):
    with open(file,'a') as f:
        f.write(msg)

def init_parameters(num_classes = 2,
                    is_training = True,
                    in_channel = 1,
                    batch_size= 30,
                    image_size= 480,
                    time_frame = 4,
                    n_threads= 10,
                    checkpoint=5,
                    q_limit=1000,
                    mse_weight=1):
    tc= ConfigObj()
    tc.num_classes = num_classes
    tc.in_channel = in_channel
    tc.is_training= is_training
    tc.batch_size = batch_size
    tc.image_size = image_size
    tc.time_frame = time_frame
    tc.n_threads = n_threads
    tc.checkpoint = checkpoint
    tc.q_limit = q_limit
    tc.lamda = mse_weight  # mse
    return tc

class randomsampler(torch.utils.data.Sampler):

    def __init__(self,index):
        self.index = index
        # self.seed = seed
        # np.random.seed(self.seed)
        np.random.shuffle(self.index)

    def __iter__(self):
        return (self.index[i] for i in range(len(self.index)))

    def __len__(self):
        return len(self.index)


if __name__ == '__main__':

    torch.manual_seed(47)
    torch.backends.cudnn.benchmark = True

    ## step1: load dataset

    train_pos_file = []
    train_neg_file = []
    valid_pos_file = []
    valid_neg_file = []

    for folder in ['E:/Network_Dataset_FIT/']:
        for i, typeid in enumerate(['TrainPositive/','TrainNegative/']):
            folderid = folder + typeid 
            for item in os.listdir(folderid):
                if item.endswith('npy'):
                    if i == 0:
                       train_pos_file.append(folderid + '/' + item)
                    elif i == 1:
                       train_neg_file.append(folderid + '/' + item)


    random.seed(23)                    
    random.shuffle(train_neg_file)
    random.shuffle(train_pos_file)
    train_config = init_parameters(num_classes = 2, in_channel = 1, time_frame = 4)
    valid_config = init_parameters(num_classes = 2, in_channel = 1, time_frame = 4, is_training=False)

    pos_set = get_data_set(in_channels = train_config.in_channel, in_frames = train_config.time_frame, image_dir = train_pos_file, lab = 1, start_frame = 0, tf_crop = True)
    neg_set = get_data_set(in_channels = train_config.in_channel, in_frames = train_config.time_frame, image_dir = train_neg_file, lab = 0, start_frame = 0, tf_crop = True)
    all_set = ConcatDataset([pos_set,neg_set])

    train_set_size = int(len(all_set)*0.9)
    valid_set_size = len(all_set) - train_set_size
    train_set, valid_set = torch.utils.data.random_split(all_set, [train_set_size, valid_set_size])
        
    print(f'Postive: {len(pos_set)}')
    print(f'Negative: {len(neg_set)}')
    weight = torch.tensor([len(pos_set)/len(train_set),len(neg_set)/len(train_set)])
    class_dict = {0:'negative',1:'positive'}
    
    print(f'Train Data: {len(train_set)}')
    print(f'Test  Data: {len(valid_set)}')

    train_data_loader = DataLoader(dataset=train_set, num_workers=train_config.n_threads,
                    batch_size=train_config.batch_size, shuffle=True, drop_last=True,
                    pin_memory=True)
    

    valid_data_loader = DataLoader(dataset=valid_set, num_workers=train_config.n_threads,
                    batch_size=train_config.batch_size, shuffle=True, drop_last=True,
                    pin_memory=True)

    ## step2: building model
    
    print('==========Building model==========')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = network.DenseNet(in_channels = train_config.in_channel, in_frames = train_config.time_frame, init_channels = 64,growth_rate = 8 ,blocks = [3,3,6,9], num_classes=2, drop_rate=0.5, bn_size = 16, batch_norm = True)
    # model depth [3,4,6,3] [3,4,23,3] [3,8,36,3]

    valid_log = 'K:/NetworkCode/log.txt'
    model.to(device)

    ## step3: define loss, optimizer and scheduler

    ce_loss = CrossEntropyLoss(weight=weight).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size = 30, gamma = 0.7)

    ## step4: define variables needed

    input = torch.Tensor(train_config.batch_size, train_config.in_channel, train_config.time_frame, train_config.image_size, train_config.image_size).to(device)
    input.requires_grad = False
    label = torch.LongTensor(train_config.batch_size).to(device)
    label.requires_grad = False

    valid_input = torch.Tensor(valid_config.batch_size, valid_config.in_channel, valid_config.time_frame, valid_config.image_size, valid_config.image_size).to(device)
    valid_input.requires_grad = False
    valid_label = torch.LongTensor(valid_config.batch_size).to(device)
    valid_label.requires_grad = False

    ## step5: start training
    print('=========start training===========')
    writer = SummaryWriter()
    niter = 300
    checkpoint = 1

    for epoch in range(niter):
        scheduler.step()
        train_ce_loss = 0
        valid_ce_loss = 0
        model.train()
        all_label = np.empty(len(train_data_loader)*train_config.batch_size,dtype = int)
        all_predicted = np.empty(len(train_data_loader)*train_config.batch_size,dtype = int)
        count = 0
        error_count = 0
        for i,batch in enumerate(tqdm(train_data_loader)):
            input.copy_(batch[0])
            label.copy_(batch[1])
            model.zero_grad()
            output,_ = model(input)
            model_ce_loss = ce_loss(output, label)
            train_ce_loss += model_ce_loss.item()
            clip_grad_norm_(model.parameters(),0.5)
            model_ce_loss.backward()
            optimizer.step()
            _,predicted = torch.max(output,1)
            for t in range(train_config.batch_size):
                all_label[count] = label[t].item()
                all_predicted[count] = predicted[t].item()
                count += 1
        c_matrix = confusion_matrix(all_label,all_predicted)
        train_ce_loss = train_ce_loss / len(train_data_loader)

        train_detect_accuracy = c_matrix[1,1]/np.sum(c_matrix[1,:])
        train_detect_specificity = c_matrix[0,0]/np.sum(c_matrix[0,:])

        train_c_matrix = c_matrix

        model.eval()
        with torch.no_grad():
            valid_losses = []
            valid_accuracies = []
            valid_specificities = []
            all_label = np.empty(len(valid_data_loader)*valid_config.batch_size,dtype = int)
            all_predicted = np.empty(len(valid_data_loader)*valid_config.batch_size,dtype = int)
            count = 0
            for i,batch in enumerate(tqdm(valid_data_loader)):
                valid_input.copy_(batch[0])
                valid_label.copy_(batch[1])
                valid_output,_ = model(valid_input)
                model_ce_loss = ce_loss(valid_output, valid_label)
                valid_ce_loss += model_ce_loss.item()
                _,valid_predicted = torch.max(valid_output,1)
                for t in range(valid_config.batch_size):
                    all_label[count] = valid_label[t].item()
                    all_predicted[count] = valid_predicted[t].item()
                    count += 1
            c_matrix = confusion_matrix(all_label,all_predicted)

            valid_ce_loss = valid_ce_loss / len(valid_data_loader)
            valid_detect_accuracy = c_matrix[1,1]/np.sum(c_matrix[1,:])
            valid_detect_specificity = c_matrix[0,0]/np.sum(c_matrix[0,:])
            print('epoch:', epoch,'loss:',valid_ce_loss,'valid_detect_sensitivity:',valid_detect_accuracy,'valid_detect_specificity:',valid_detect_specificity,'\n')
            print('epoch:', epoch,'loss:',train_ce_loss,'training_detect_sensitivity:',train_detect_accuracy,'training_detect_specificity:',train_detect_specificity,'\n')
            text = ('Epoch: %d, Train_Loss: %.3f, Valid_Loss: %.3f, Train_sensitivity: %.3f, Train_specificity: %.3f, Valid_sensitivity: %.3f, Valid_specificity: %.3f \n' %
                (epoch, train_ce_loss, valid_ce_loss, train_detect_accuracy, train_detect_specificity, valid_detect_accuracy, valid_detect_specificity))
            print_and_save_msg(text, valid_log)

            valid_losses.append(valid_ce_loss)
            valid_accuracies.append(valid_detect_accuracy)
            valid_specificities.append(valid_detect_specificity)
        
        if epoch%checkpoint == 0:
            torch.save(model.state_dict(),f'K:/NetworkCode/model_latest/model_detection_epoch{epoch}.pth')
        
        writer.add_scalars(f'detection/loss',{'train':train_ce_loss,'valid':valid_losses[0]},epoch)
        writer.add_scalars(f'detection/{class_dict[1]}',{'train':train_detect_accuracy,'valid':valid_accuracies[0]},epoch)
        writer.add_scalars(f'detection/{class_dict[0]}',{'train':train_detect_specificity,'valid':valid_specificities[0]},epoch)