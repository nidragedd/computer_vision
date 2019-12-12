# Computer Vision experiments
Goal of this project is to explore Computer Vision with OpenCV and Python. A lot of this work is based on very good tutorials
proposed by Dr. Adrian Rosebrock on his wonderful blog [pyimagesearch](https://www.pyimagesearch.com/).

## Table of contents
* [1. Technical Section](#1-technical-section)
* [2. Context - Objectives](#2-context---objectives)
* [3. Object Detection](#3-object-detection)

---

## 1. TECHNICAL SECTION
### Dependencies & Installation - Create your CONDA virtual environment
Easiest way is to create a virtual environment through **[conda](https://docs.conda.io/en/latest/)**
and the given `environment.yml` file by running this command in a terminal (if you have conda, obviously):
```
conda env create -f environment.yml
```

If you do not have/want to use conda for any reason, you can still setup your environment by running some `pip install`
commands. Please refer to the `environment.yml` file to see what are the dependencies you will need to install.  
Basically, this project requires **Python 3.7** in addition to common image manipulation packages (such as 
[opencv 4.x](https://opencv.org/) (note that [numpy](https://www.numpy.org/) will be required), [PIL](https://pillow.readthedocs.io/en/stable/)).

There are those additional packages in order to expose our work within a webapp:
* [Flask](https://palletsprojects.com/p/flask/): used as web application framework/engine to run the app over HTTP

### Directory & code structure
Here is the structure of the project:
```
    project
      |__ models    (contains downloaded pre-trained models)
      |__ src       (python modules and scripts)
            |__ detection   (scripts called to perform object or motion detection)
                    |__ models      (contains models zoo for object & motion detection)
                    |__ object      (contains scripts for object detection in pictures with pre-trained models)
            |__ motion   (scripts called to do read & stream live motion detection)
            |__ static      (HTML static resources to serve through Flask)
            |__ templates   (HTML templates to serve through Flask, this is the 'view' part)
            |__ utils       (python helper & utility functions)
            |__ webapp      (python files corresponding to the webapp: this is the 'controller' part)
            |__ app.py      (script called to run the Flask server)
            |__ main_xxx.py (main scripts to test a specific functionality)
```

### Run the app on your local computer
1. Run the following command in the project's root directory to run the web app.
    `python src/app.py -i 0.0.0.0 -p 3001`
Then go to http://0.0.0.0:3001/ or [http://localhost:3001/](http://localhost:3001/) with your favorite browser

2. You can launch specific functionality with some `main_xxx` python scripts. They are all under project's src directory.
***PAY ATTENTION*** to mandatory arguments if any (have a look at the argparse configuration)

---
## 2. CONTEXT & OBJECTIVES
As a regular reader of Adrian Rosebrock [pyimagesearch blog](https://www.pyimagesearch.com/) or [learnopencv](https://www.learnopencv.com/), I wanted to give a try to some of the great tutorials
but in such a way that everything remains available in a single place and, if and when possible, with factorized code.  
That is why a lot of what you can read in this project comes from there (plus some additional research).  
I have made the effort to propose the different functions within a webapp that you can run on your computer.

---
## 3. OBJECT DETECTION
### Resources
* [pyimagesearch: object detection with SSD300](https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv)
* [pyimagesearch: object detection with YOLO](https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv)
* [towardsdatascience: review of what is a Single Shot Detector](https://towardsdatascience.com/review-ssd-single-shot-detector-object-detection-851a94607d11)
* [Another medium post on SSD multibox detection](https://medium.com/@jonathan_hui/ssd-object-detection-single-shot-multibox-detector-for-real-time-processing-9bd8deac0e06)
* [AI Google blog: review of MobileNetV3](https://ai.googleblog.com/2019/11/introducing-next-generation-on-device.html)


### Download models
Pre-trained models are heavy files so they are not committed within this repository. Here are the links you can use to download them:
| Model link | Github project link | Type | Network base | Input image size | Dataset trained on | mAP |
|------------|---------------------|------|--------------|------------------|--------------------|-----|
| [Link](https://drive.google.com/open?id=0B3gersZ2cHIxRm5PMWRoTkdHdHc)     | [Link](https://github.com/chuanqi305/MobileNet-SSD)           | SSD  | MobileNet v3   | 300  | PASCAL VOC0712 | 72.7% |
| [Link](https://drive.google.com/file/d/0BzKzrI_SkD1_dUY1Ml9GRTFpUWc/view) | [Link](https://github.com/weiliu89/caffe/tree/ssd)            | SSD  | VGG16          | 300* | MSCOCO         | 25.1% |
| [Link](https://drive.google.com/file/d/0BzKzrI_SkD1_dlJpZHJzOXd3MTg/view) | [Link](https://github.com/weiliu89/caffe/tree/ssd)            | SSD  | VGG16          | 512* | MSCOCO         | 28.8% |
| [Link](https://drive.google.com/file/d/0BzKzrI_SkD1_a2NKQ2d1d043VXM/view) | [Link](https://github.com/weiliu89/caffe/tree/ssd)            | SSD  | VGG16          | 300* | ILSVRC16       | --    |
| [Link](https://drive.google.com/file/d/0BzKzrI_SkD1_X2ZCLVgwLTgzaTQ/view) | [Link](https://github.com/weiliu89/caffe/tree/ssd)            | SSD  | VGG16          | 500* | ILSVRC15       | --    |

_Note:_: SSD300* and SSD512* are models that trained with data augmentation ([source](https://arxiv.org/pdf/1512.02325v4.pdf))


### Notes
* Single Shot Detectors (SSDs) and YOLO use a one-stage detector strategy.
* MobileNets are designed for resource constrained devices (raspberry, phone).
* YOLO is faster but less accurate than SSD, mostly with small objects.
* MobileNets differ from traditional CNNs through the usage of depthwise separable convolution in which we split convolution into two stages (to dramatically
reduce the number of parameters in our network, see [this good explanation](https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728)):
    * A 3×3 depthwise convolution.
    * Followed by a 1×1 pointwise convolution.
