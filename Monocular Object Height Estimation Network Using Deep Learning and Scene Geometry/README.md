# MOHE Net
Estimating the heights of objects in the field of view has applications in many tasks such as robotics, autonomous platforms and video surveillance. Object height is a concrete and indispensable characteristic people or machine could learn and capture. Many actions such as vehicle avoiding obstacles will be taken based on it. Traditionally, object height can be estimated using laser ranging, radar or stereo camera. Depending on the application, cost of these techniques may inhibit their use, especially in autonomous platforms. Use of available sensors with lower cost would make the adoption of such techniques at higher rates. Our approach to height estimation requires only a single 2D image. To solve this problem we introduce the Monocular Object Height Estimation Network (MOHE-Net).

# Requirements
Python 3.8 or later with all requirements.txt dependencies installed, including `torch>=1.7`. To install run:
```bash
$ pip install -r requirements.txt
```

# Pretrained weights
YOLOv5 pretrained weights are available at [`one drive`](https://buckeyemailosu-my.sharepoint.com/:f:/r/personal/wei_909_buckeyemail_osu_edu/Documents/YOLOv5%20Pre-trained%20Models?csf=1&web=1&e=AUQf3e). Please download and save them into `model\yolov5\weights`.


# Inference
```bash
$ pip install -r requirements.txt
```
