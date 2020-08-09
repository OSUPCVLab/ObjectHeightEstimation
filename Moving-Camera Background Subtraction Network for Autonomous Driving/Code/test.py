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
    data_augmentations = Compose([CenterCrop([2050, 2560]), UpCrop([960,2560]), CenterCrop([960,1600]), Scale(400)])  # H*W 200*320

    testDL = ImagesDataset(opt.image_dir,
                           opt.label_dir,
                           type=opt.type,
                           augmentations=data_augmentations,
                           return_name=True)

    test_loader = data.DataLoader(testDL,
                                 batch_size=opt.batch_size,
                                 shuffle=False,
                                 drop_last=True)

    # BiSeNet Model
    out_channel = 2
    net = model.BiSeNet(out_channel, opt.backbone, Deconvolution=opt.Deconvolution).to(device)


    # Load trained weights
    net.load_state_dict(torch.load('./output/model/{}/{}_epoch_{}.pt'.format(opt.backbone, opt.backbone, opt.epoch)))
    print('Successfully load model weights!\n')


    # specify loss function
    # criterion = nn.CrossEntropyLoss().to(device)
    alpha = 0.75
    criterion_p = FocalLoss(gamma=0.5, alpha=alpha)


    # DenseCRF
    post_processor = crf.xyt_DenseCRF(
        iter_max=10,    # iteration times
        scale=0.9,  # network output confidence ranging (0,1)
        pos_w=0.5,  # gaussian kernel weight
        pos_xy_std=(2,2),   # position std
        bi_w=3.5,   # bilateral weight
        bi_xy_std=(2,2),  # position std, larger means ignorance
        # bi_t_std = tuple(np.logspace(0,opt.crf_num-1,opt.crf_num, base=2))  # t-1 mask std
        bi_t_std = tuple(range(1, opt.crf_num+1))  # t-1 mask std
        )

    print('Start inference...')
    ####################
    ####### Test #######
    ####################

    test_loss, iou_score, crf_loss, crf_iou_score, crf_times = [], [], [], [], []
    name_loss = {}
    img_names = []
    previous_all_masks = np.empty([0,240,400], dtype=int)
    total_num = 0
    with torch.no_grad():

        for imgs, gts, imgs_name, gts_name in tqdm(test_loader, ascii=True, desc='Testing'):

            imgs, gts = imgs.to(device), gts.to(device)

            outputs = net(imgs)
            probs = nn.functional.softmax(outputs, dim=1) # shape = [batch_size, C, H, W]
            predictions = np.argmax(probs.detach().cpu().numpy(), axis=1).astype(int)
            previous_all_masks = np.concatenate((previous_all_masks, predictions), axis=0)


            # compute test mIoU Loss
            for i in range(opt.batch_size):
                iou_score.append(compute_iou_score(np.argmax(probs[i].detach().cpu().numpy(), axis=0).astype(int), gts[i].cpu().numpy(), opt.class_))
                img_names.append(imgs_name[i])
        
            loss = criterion_p(outputs, gts)
            test_loss.append(loss.item())


            for i in range(opt.batch_size):
                if opt.use_crf:# is post process
                
                    # prepare previous n masks
                    if total_num*opt.batch_size+i<opt.crf_num:
                        previous_num_masks = (np.expand_dims(previous_all_masks[total_num*opt.batch_size+i],axis=2)).repeat(opt.crf_num, axis=2)
                    else:
                        previous_num_masks = np.expand_dims(previous_all_masks[total_num*opt.batch_size+i-1], axis=2)
                        for j in range(2,opt.crf_num+1):
                            previous_num_masks = np.concatenate((previous_num_masks, np.expand_dims(previous_all_masks[total_num*opt.batch_size+i-j], axis=2)), axis=2)

                    raw_image = previous_num_masks.astype(np.uint8)
                    start_time = time.time()
                    prob = post_processor(raw_image, probs[i].detach().cpu().numpy())
                    crf_times.append(time.time() - start_time)
                    pred = np.argmax(prob, axis=0).astype(int)
                    crf_loss.append(criterion_p(torch.unsqueeze(torch.from_numpy(prob),0).to(device), torch.unsqueeze(gts[i],0)).item())
                    crf_mean_iou = compute_iou_score(pred, gts[i].cpu().numpy(), opt.class_)
                    crf_iou_score.append(crf_mean_iou)
                    name_loss[imgs_name[i]] = crf_mean_iou

                # save predicted frames
                name = './output/'+opt.backbone+'/'+opt.type+gts_name[i].split(opt.type)[-1]
                file_name = name.split('Camera 5')[0]+'Camera 5'

                if not os.path.exists(file_name):
                    os.makedirs(file_name)
                
                if not opt.use_crf:
                    pred = (np.expand_dims(torch.argmax(F.softmax(outputs[i].cpu(), dim=0), dim=0).numpy(), axis=2)).repeat(3,axis=2)
                else:
                    pred = (np.expand_dims(pred, axis=2)).repeat(3,axis=2)
                prediction = (np.expand_dims(predictions[i], axis=2)).repeat(3,axis=2)

                target = (np.expand_dims(gts[i].cpu().numpy(), axis=2)).repeat(3,axis=2)
                img = np.transpose(imgs[i].cpu().numpy(),(1,2,0))[:, :, ::-1]

                prediction, pred, target = 0.8*prediction, 0.7*pred, 0.6*target
                target_pred = np.concatenate((img,prediction, pred, target),axis=1)*255  # prediction w/o CRF and pred w/ CRF
                
                cv2.imwrite(name, target_pred)

            # free gpu memory
            imgs, gts = imgs.detach(), gts.detach()
            imgs, gts = None, None
            
            total_num+=1


        # plot test loss
        if opt.use_crf:
            p1, = plt.plot(iou_score, color='blue')
            p2, = plt.plot(crf_iou_score, color='red')
            plt.legend((p1, p2), ('mIoU', 'mIoU with CRF'), loc=1)
        else:
            plt.plot(iou_score)
        plt.title('IoU score')
        plt.savefig('./output/plot/{}/IoU_score_epoch_{}.png'.format(opt.backbone, opt.epoch))
        plt.close()
        
        # plot mIoU loss difference
        if opt.use_crf:
            plt.plot([crf_iou_score[i]-iou_score[i] for i in range(len(iou_score))])
            plt.title('crf_iou_score - iou_score')
            plt.savefig('./output/plot/{}/iou_diff.png'.format(opt.backbone))
            plt.close()
        
        # print test results
        if opt.use_crf:
            print('dense crf time = %s' % (np.mean(crf_times)))
            print('mIoU score using crf is {:.2f}%'.format(np.mean(crf_iou_score)*100))
            print('mIoU score not using crf is {:.2f}%'.format(np.mean(iou_score)*100))
            print('\n')
            print('CrossEntropyLoss using crf is {:.5f}'.format(np.mean(crf_loss)))
            print('CrossEntropyLoss not using crf is {:.5f}'.format(np.mean(test_loss)))
            # for i in range(len(img_names)):
                # if crf_iou_score[i]-iou_score[i]>0.10:
                    # print(img_names[i], crf_iou_score[i]-iou_score[i])
        else:
            print('mIoU score not using crf is {:.2f}%'.format(np.mean(iou_score)*100))
            print('\n')
            print('CrossEntropyLoss not using crf is {:.2f}'.format(np.mean(test_loss)))



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
    
    parser.add_argument("--class_", type=int, default=0, help="compute mIoU of background or non-background")
    
    parser.add_argument("--epoch", type=int, default=0, help="test epoch selection")
    
    parser.add_argument('--use_crf', default=False, action='store_true', help='use crf or not')
    
    parser.add_argument("--crf_num", type=int, default=1, help="test epoch selection")

    opt = parser.parse_args()
    
    main(opt)