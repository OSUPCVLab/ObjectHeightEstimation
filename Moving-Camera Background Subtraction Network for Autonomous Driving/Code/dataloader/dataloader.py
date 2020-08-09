import os
from imageio import imread
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

    
class ImagesDataset(Dataset):
    def __init__(self,
                image_dir,
                label_dir,
                type='train',
                augmentations=None,
                is_transform=True,
                compute_mean_std=False,
                img_norm=True,
                return_name=False,
                ):
        self.image_dir = self.load_img(image_dir, type)
        self.label_dir = self.load_lbl(label_dir, type)
        self.type = type
        self.is_transform = is_transform
        self.compute_mean_std = compute_mean_std    # compute dataset mean and std
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.return_name = return_name

    def __getitem__(self,index):
        assert len(self.image_dir)==len(self.label_dir)
        image_path = self.image_dir[index]
        label_path = self.label_dir[index]
        
        # compute mean and std of training dataset
        if self.compute_mean_std:
            self.Mean, self.Std = self.compute_mean_and_std(self.image_dir)
        elif self.type=='train':
            self.Mean, self.Std = [0, 0, 0], [1, 1, 1]  # train dataset mean and std
            #self.Mean, self.Std = [0.292, 0.314, 0.330], [0.231, 0.245, 0.247]  # train dataset mean and std
        elif self.type=='val':
            self.Mean, self.Std = [0, 0, 0], [1, 1, 1]  # train dataset mean and std
            #self.Mean, self.Std = [0.313, 0.334, 0.347], [0.237, 0.247, 0.248]  # val dataset mean and std
        elif self.type=='test':
            self.Mean, self.Std = [0, 0, 0], [1, 1, 1]  # train dataset mean and std
            #self.Mean, self.Std = [0.304, 0.322, 0.334], [0.241, 0.252, 0.251]  # test dataset mean and std
        else:
            print('Please provide correct dataset type!')
        
        img = np.asarray(Image.open(image_path))
        lbl = np.asarray(imread(label_path))

        # Apply data augmentations to original images and gts
        if self.augmentations:
            img, lbl = self.augmentations(img, lbl)

        # regularize image values to mean=0 and std=1 distribution
        if self.is_transform:
            img, lbl = self.transforms(img, lbl)

        if self.return_name:       # return names for saving predicted labels to output dataset
            return img, lbl, image_path, label_path
        else:
            return img, lbl
        
    def __len__(self):
        return len(self.image_dir)
        
    def transforms(self, img, lbl):
        img = np.array(Image.fromarray(img))
        lbl = np.array(Image.fromarray(lbl))

        img = img.astype(np.float64)

        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            img = img.astype(float) / 255.0
            img = (img-self.Mean)/self.Std

        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        # To Tensor
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl
    
    # search all train/val/test frames
    def load_img(self, image_dir, type):
        img_dirs = []
        for record in os.listdir(image_dir+type):
            for img_name in os.listdir(image_dir+type+'/'+record+'/'+'Camera 5'):
                img_dirs.append(image_dir+type+'/'+record+'/'+'Camera 5'+'/'+img_name)

        return img_dirs
    
    # search all train/val/test gts
    def load_lbl(self, label_dir, type):
        lbl_dirs = []
        for record in os.listdir(label_dir+type):
            for lbl_name in os.listdir(label_dir+type+'/'+record+'/'+'Camera 5'):
                lbl_dirs.append(label_dir+type+'/'+record+'/'+'Camera 5'+'/'+lbl_name)

        return lbl_dirs
    
    # compute dataset train/val/test mean and std
    def compute_mean_and_std(self, image_path):
        X = np.empty([len(image_path),240,400,3])   # image size needs to be mannually set
        for i, path in enumerate(image_path):
            img = np.asarray(Image.open(path))
            if self.augmentations:
                img, _ = self.augmentations(img, img[:,:,0])
            X[i] = img/255.0
            print('\r{:.2f}% finished!'.format((i+1)/len(image_path)*100), end=' ')
        # calculate the mean and std along the (0, 1) axes
        X_mean = np.mean(X, axis=(0, 1, 2))
        X_std = np.std(X, axis=(0, 1, 2))
        # the the mean and std
        print(X_mean, X_std)
        
        return X_mean, X_std