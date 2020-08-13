# MBSnet
This project aims to provide fast, accurate pixelwise semantic segmentation for moving-camera background subtraction using PyTorch.

# Software Installment
* [PyTorch 1.6.0](https://pytorch.org/)
* [opencv 4.4.0.40](https://pypi.org/project/opencv-python/)
* [tmdq 4.48.2](https://pypi.org/project/tqdm/)

# Training
1. Download datasets and add local dir for training and validation images and labels.
2. Modify `parser.add_argument` in `train.py` according to your requirements. Arguments include:
* --image_dir
* --label_dir
* --batch_size
* --backbone
* --start_epoch
* --Deconvolution

3. Train MBSNet
