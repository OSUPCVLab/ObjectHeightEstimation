import os
import argparse
import warnings
warnings.filterwarnings("ignore") 
from tqdm import tqdm
import time

import numpy as np
from matplotlib import pyplot as plt
import cv2

from utils.augmentations import *
from dataloader.dataloader import ImagesDataset
from model import model
from loss.loss import OhemCELoss, SoftmaxFocalLoss, FocalLoss

import torch
from torch.utils import data
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torchvision.transforms as transforms

def compute_iou_score(prediction, target, c):
    W, H = prediction.shape
    prediction, target = prediction.reshape(W*H), target.reshape(W*H)
    prediction_c = np.where(prediction==c)
    target_c = np.where(target==c)
    intersection = np.intersect1d(prediction_c, target_c)
    union = np.union1d(prediction_c, target_c)
    if np.sum(union)==0:
        return np.sum(intersection)
    return np.sum(intersection) / np.sum(union)


def main(opt):
    # device configuration
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

    # load training dataset and test dataset
    crop_size = [240, 400]
    data_augmentations = Compose([CenterCrop([2050, 2560]), UpCrop([960,2560]), CenterCrop([960,1600]), Scale(crop_size[1])])  # H*W 240*400

    trainDL = ImagesDataset(opt.image_dir,
                            opt.label_dir,
                            compute_mean_std = False,
                            augmentations=data_augmentations
                            )

    valDL = ImagesDataset(opt.image_dir,
                          opt.label_dir,
                          type='test',
                          compute_mean_std = False,
                          augmentations=data_augmentations
                          )

    train_loader = data.DataLoader(trainDL,
                                   batch_size=opt.batch_size,
                                   shuffle=True,
                                   drop_last=True)

    val_loader = data.DataLoader(valDL,
                                 batch_size=opt.batch_size,
                                 shuffle=False,
                                 drop_last=True)


    # MBSNet Model
    out_channel = 2
    net = model.MBSNet(out_channel, opt.backbone, Deconvolution=opt.Deconvolution).to(device)


    # specify start epoch
    if opt.start_epoch!=0:
        net.load_state_dict(torch.load('./output/model/{}/{}.pt'.format(opt.backbone, opt.backbone)))
        print('Successfully load model weights!')

    # specify loss function and optimizer
    alpha = 0.75
    gamma = 0.5
    criterion_p = FocalLoss(gamma=gamma, alpha=alpha)
    criterion_16 = FocalLoss(gamma=gamma, alpha=alpha)
    criterion_32 = FocalLoss(gamma=gamma, alpha=alpha)
    
    optimizer = optim.Adam(net.parameters(), lr = 1e-3)
    # optimizer = optim.SGD(net.parameters(),
                          # lr = 2e-3,
                          # momentum=0.9,
                          # weight_decay=5e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    # specify initialized parameters
    total_epochs = 80
    val_loss_min = 0

    print('Start Training...')

    for epoch in range(opt.start_epoch, total_epochs):
        print('Epoch: {} \tLearning Rate:{:.3e}'.format(epoch, scheduler.get_lr()[0]))

        # Initialize time and loss
        start_time = time.time()
        train_loss = []
        val_loss = []

        # Clear cuda cache
        torch.cuda.empty_cache()


        ########################
        ####### Training #######
        ########################

        net.train()
        for imgs, gts in tqdm(train_loader, ascii=True, desc="Training"):
            imgs, gts = imgs.to(device), gts.to(device)

            optimizer.zero_grad()
            out_p, out_16, out_32 = net(imgs, return_aux=True)

            loss_p = criterion_p(out_p, gts)
            loss_16 = criterion_16(out_16, gts)
            loss_32 = criterion_32(out_32, gts)
            loss = loss_p + loss_16 + loss_32
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

            #free gpu memory
            imgs, gts = imgs.detach(), gts.detach()
            imgs, gts = None, None

        # Update learning rate
        scheduler.step()

        ########################
        ###### Validation ######
        ########################

        with torch.no_grad():
            for imgs, gts in tqdm(val_loader, ascii=True, desc='Validation'):
                imgs, gts = imgs.to(device), gts.to(device)

                out_p = net(imgs)
                probs = nn.functional.softmax(out_p, dim=1) # shape = [batch_size, C, H, W]
                for i in range(opt.batch_size):
                    val_loss.append(compute_iou_score(np.argmax(probs[i].detach().cpu().numpy(), axis=0).astype(int), gts[i].cpu().numpy(), 0))

                # free gpu memory
                imgs, gts = imgs.detach(), gts.detach()
                imgs, gts = None, None

        # Save trained model
        torch.save(net.state_dict(), './output/model/{}/{}_epoch_{}.pt'.format(opt.backbone, opt.backbone, epoch))

        # plot training and validation loss
        plt.plot(train_loss)
        plt.plot(val_loss)
        plt.title('Loss')
        plt.legend(['Train loss', 'Validation loss'], loc=1)
        plt.savefig('./output/plot/{}/Epoch_{}.png'.format(opt.backbone, epoch))
        plt.close()

        print('Training loss is {:.4f}\tValidation mIoU is {:.4f}'.format(np.mean(train_loss), np.mean(val_loss)))
        if np.mean(val_loss)>val_loss_min:
            print('mIoU Increases {:.4f} --> {:.4f}, saving model...'.format(val_loss_min, np.mean(val_loss)))
            val_loss_min = np.mean(val_loss)
            torch.save(net.state_dict(), './output/model/{}/{}_epoch_{}_best.pt'.format(opt.backbone, opt.backbone, epoch))
        else:
            print('mIoU keeps {:.4f}, not saving model...'.format(val_loss_min))

        print('Running time is {:.2f} minutes\n'.format((time.time()-start_time)/60))

    print('Training ends...')


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_dir", type=str, default="E:\\Datasets\\Jianli\\ApolloScape Dataset\\Image\\",
                        help="Relative path to the directory containing images to detect.")

    parser.add_argument("--label_dir", type=str, default="E:\\Datasets\\Jianli\\ApolloScape Dataset\\Label\\",
                        help="Relative path to the directory containing labels to detect.")

    parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")

    parser.add_argument("--backbone", type=str, default='Resnet101', help="Training backbone of MBSNet model")

    parser.add_argument("--start_epoch", type=int, default=0, help="Training start epoch")

    parser.add_argument('--Deconvolution', default=False, action='store_true', help='use Deconvolution or Bilinear Interpolation to achieve 8x upsampling')

    opt = parser.parse_args()
    main(opt)