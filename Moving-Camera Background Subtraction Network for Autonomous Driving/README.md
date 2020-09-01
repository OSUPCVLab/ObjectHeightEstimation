# MBSNet
This project aims to provide fast, accurate pixelwise semantic segmentation for moving-camera background subtraction using PyTorch.

# Software Installment
* [PyTorch 1.6.0](https://pytorch.org/)
* [opencv 4.4.0.40](https://pypi.org/project/opencv-python/)
* [tmdq 4.48.2](https://pypi.org/project/tqdm/)
* [pydensecrf](https://github.com/lucasb-eyer/pydensecrf)

# Training
1. Download [`datasets`](https://github.com/OSUPCVLab/Ford2020/tree/master/Moving-Camera%20Background%20Subtraction%20Network%20for%20Autonomous%20Driving/Dataset) and add local dir for training and validation images and labels.
2. Modify `parser.add_argument` in `train.py` according to your requirements. Input arguments include:
* `--image_dir`
* `--label_dir`
* `--batch_size`
* `--backbone`
* `--start_epoch`
* `--Deconvolution`

Descriptions about above arguments are available on `train.py`.

3. Train MBSNet

# Inference
Inference [`datasets`](https://github.com/OSUPCVLab/Ford2020/tree/master/Moving-Camera%20Background%20Subtraction%20Network%20for%20Autonomous%20Driving/Dataset) are also available. Please assign a GPU for inference if available. `speed_analysis.py` is written for testing trained MBSNet inferencing speed.
1. Add local testing images and labels dir.
2. Modify `parser.add_argument` in `test.py` according to your requirements. Besides those input arguments listed above `train.py`, `test.py` also includes:
* `--type`
* `--class_`
* `--epoch`
* `--use_crf`
* `--crf_num`

3. Trained MBSNet under Xception39 basemodel is available [`OneDrive`](https://buckeyemailosu-my.sharepoint.com/:f:/r/personal/wei_909_buckeyemail_osu_edu/Documents/Pre-trained%20Models?csf=1&web=1&e=kFdfGo).

# Disclaimer
This project is an ongoing project. Future work will be released. Feel free to open an issue if you get stuck anywhere.

# Contact
Jianli Wei

wei.909@osu.edu

[PCV Lab, The Ohio State University](https://pcvlab.engineering.osu.edu/)
