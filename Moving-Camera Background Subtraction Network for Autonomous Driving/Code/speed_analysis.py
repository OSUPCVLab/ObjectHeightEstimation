import os
import argparse
import warnings
warnings.filterwarnings("ignore") 
from tqdm import tqdm
import time

import numpy as np
from matplotlib import pyplot as plt
import cv2
#import pretty_errors

from utils.augmentations import *
from utils import crf
from dataloader.dataloader import ImagesDataset
from model import model, model_update
from loss.loss import OhemCELoss, SoftmaxFocalLoss, FocalLoss

import torch
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


def main(opt):
    # device configuration
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

    # load training dataset and test dataset
    data_augmentations = Compose([CenterCrop([2050, 2560]), UpCrop([960,2560]), CenterCrop([960,1600]), Scale(400)])  # H*W 200*320

    testDL = ImagesDataset(opt.image_dir,
                           opt.label_dir,
                           type=opt.type,
                           augmentations=data_augmentations,
                           return_name=True,
                           return_previous_label=True)

    test_loader = data.DataLoader(testDL,
                                 batch_size=opt.batch_size,
                                 shuffle=False,
                                 drop_last=True)

    # BiSeNet Model
    out_channel = 2
    net = model_update.BiSeNet(out_channel, opt.backbone, Deconvolution=opt.Deconvolution).to(device)


    # Load trained weights
    net.load_state_dict(torch.load('./output/model/{}/{}_epoch_{}.pt'.format(opt.backbone, opt.backbone, opt.epoch)))
    print('Successfully load model weights!\n')


    # DenseCRF
    post_processor = crf.xyt_DenseCRF(
        iter_max=10,    # iteration times
        scale=0.9,  # network output confidence ranging (0,1)
        pos_w=0.5,  # gaussian kernel weight
        pos_xy_std=(2,2),   # position std
        bi_w=3.5,   # bilateral weight
        bi_xy_std=(2,2),  # position std, larger means ignorance
        bi_t_std=2, # t-1 mask std
    )

    print('Start inference...')
    ####################
    ####### Test #######
    ####################

    nn_time = []
    crf_time = []
    with torch.no_grad():

        for imgs, gts, labels, imgs_name, gts_name in test_loader:
            start_time = time.time()
        
            imgs, gts = imgs.to(device), gts.to(device)

            outputs = net(imgs)
            probs = nn.functional.softmax(outputs, dim=1) # shape = [batch_size, C, H, W]

            nn_time.append(time.time()-start_time)

            if opt.use_crf:# is post process
                for i in range(opt.batch_size):
                    raw_image = labels[i].numpy().astype(np.uint8)
                    prob = post_processor(raw_image, probs[i].detach().cpu().numpy())
                    pred = np.argmax(prob, axis=0).astype(int)
                crf_time.append(time.time()-start_time)
            
            # free gpu memory
            imgs, gts = imgs.detach(), gts.detach()
            imgs, gts = None, None
        
        # time evaluation
        print('Inferring {:.1f} frames per second...'.format(opt.batch_size*len(test_loader)/np.sum(nn_time)))
        if opt.use_crf:
            print('Inferring {:.1f} frames per second with CRF...'.format(opt.batch_size*len(test_loader)/np.sum(crf_time)))


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_dir", type=str, default="E:\\Datasets\\Jianli\\ApolloScape Dataset\\Image\\",
                        help="Relative path to the directory containing images to detect.")

    parser.add_argument("--label_dir", type=str, default="E:\\Datasets\\Jianli\\ApolloScape Dataset\\Label\\",
                        help="Relative path to the directory containing labels to detect.")

    parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")

    parser.add_argument("--backbone", type=str, default='Resnet101', help="Training backbone of BiSeNet model")
    
    parser.add_argument('--Deconvolution', default=False, action='store_true', help='use Deconvolution or Bilinear Interpolation to achieve 8x upsampling')
    
    parser.add_argument('--type', type=str, default='test', help='type of dataset')
    
    parser.add_argument("--epoch", type=int, default=0, help="test epoch selection")
    
    parser.add_argument('--use_crf', default=False, action='store_true', help='use crf or not')

    opt = parser.parse_args()
    
    main(opt)