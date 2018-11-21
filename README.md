# Curve-Text-Detector

Released on December 06, 2017

# Recent Update 

(July 25, 2018)

- We provide a brief evaluation script for researches to evaluate their own methods on the CTW1500 dataset. The instruction and details are given in tools/ctw1500_evaluation/Readme.md.

- The results from recent papers on the CTW1500 dataset are listed (Note that the table only listed the results from published papers. If your detection results on this dataset is missed, please let me know. )

| Method     |  Recall (%)  |  Precision (%)  |   Hmean (%)     |   Extra Data    |
|:--------:  | :-----:   | :----:      |  :-----:     |    :-----:    |
|Proposed CTD [[paper]](https://arxiv.org/abs/1712.02170)     | 65.2     |  74.3       |    69.5      |      -         |
|Proposed CTD+TLOC [[paper]](https://arxiv.org/abs/1712.02170)| 69.8     |  74.3       |    73.4      |       -        |
|SLPR [[paper]](https://arxiv.org/abs/1801.09969)             | 70.1     |  80.1       |    74.8      |        -       |
|TextSnake [[paper]](https://arxiv.org/abs/1807.01544)        | 85.3     |  67.9       |    75.6     |          PT(S(800K))    |
|PSENet-4s [[paper]](http://export.arxiv.org/abs/1806.02559)  |  78.13    |  85.49      |    79.29     |   PT(IC15(1K)+ MLT(9K))     |
|PSENet-2s [[paper]](http://export.arxiv.org/abs/1806.02559)  | 79.3     |  81.95      |    80.6     |   PT(IC15(1K)+ MLT(9K))          |
|PSENet-1s [[paper]](http://export.arxiv.org/abs/1806.02559)  | 79.89    |  82.50       |    81.17     |   PT(IC15(1K)+ MLT(9K))          |


*PT: pretrained data. S: synthesize data. PA: private data. PB (or name): public data.

# Description

Curved text are very common in our real-world. For examples, text in most kinds of columnar objects (bottles, stone piles, etc.), spherical objects, plicated plane (clothes, streamer, etc.), coins, logos, signboard and so on. Current datasets have very little curved text, and it is defective to label such text with quadrangle let alone rectangle. Curved bounding box has three remarkable advantages:
<div align=center><img src="images/1-1.jpg" width="50%" ></div>
<div align=center><img src="images/1-2.jpg" width="50%" ></div>
<div align=center><img src="images/1-3.jpg" width="50%"></div>

* Avoid needless overlap

* Less background noise

* Avoid multiple text lines

# Clone the Curve-Text-Detector repository

Clone the Curve-Text-Detector repository
  ```Shell
  git clone https://github.com/Yuliang-Liu/Curve-Text-Detector.git --recursive
  ```

# Getting Started
## Dataset

<img src="images/annotation.jpg" width="100%">

The SCUT-CTW1500 dataset can be downloaded through the following link:

(https://pan.baidu.com/s/1eSvpq7o PASSWORD: fatf) (BaiduYun. Size = 842Mb)

or (https://1drv.ms/u/s!Aplwt7jiPGKilH4XzZPoKrO7Aulk) (OneDrive)

unzip the file in ROOT/data/ 

### Dataset Information

a) Train/ - 1000 images.

b) Test/ - 500 images.

c) Each image contains at least 1 curved text.

The visualization of the annotated images can be downloaded through the following link:

(https://pan.baidu.com/s/1eR641zG PASSWORD: 5xei) (BaiduYun. Size = 696 Mb).

## Pre-trained model and our trained model

We use resnet-50 model as our pre-trained model, which can be download through the following link:

(https://pan.baidu.com/s/1eSJBL5K PASSWORD: mcic) (Baidu Yun. Size = 102Mb)

or (https://1drv.ms/u/s!Aplwt7jiPGKilHwMsW2N_bfnb0Bx) (OneDrive)

put model in ROOT/data/imagenet_models/

Our model trained with SCUT-CTW1500 training set can be download through the following link:

(https://pan.baidu.com/s/1gfs5vH5 PASSWORD: 1700) (BaiduYun. Size = 114Mb)

or (https://1drv.ms/u/s!Aplwt7jiPGKilH0rLDFrRof8qmRD) (OneDrive)

put model in ROOT/output/

* [test.sh](./test.py) Downloading the dataset and our ctd_tloc.caffemodel, and running this file to evaluate our method on the SCUT-CTW1500 test set. Uncommend --vis to visualize the detecting results.

* [my_train.sh](./my_train.sh) This file shows how to train on the SCUT-CTW1500 dataset. Downloading the dataset and resnet-50 pre-trained model, and running my_train.sh to start training. 

Both train and test require less than 4 GB video memory.

* [demo.py](./tools/demo.py) (cd tools/) then (python demo.py). This file easily shows how to test other images. With provided model, it can produce like

<div align=center><img src="images/demo_result.png" width="50%" ></div>

# Comparing smooth effect by TLOC 
Train and test files are put under (model/ctd/smooth_effect/), and both the training and testing procedures are the same as above.

To visulize the ctd+tloc, simply uncomment ctd in the last of the test.prototxt, vice versa. Below are the first three images in our test set: 

<table><tr>
    <td><img src="images/s1.png" width="240" height="180" border=0></td>
    <td><img src="images/s2.png" width="240" height="180" border=0></td>
    <td><img src="images/s3.png" width="240" height="180" border=0></td>
</tr></table>

If you are insterested in it, you can train your own model to test. Because training doesn't require so much time, we don't upload our own model (Of course, you can email me for it). 

# Long side interpolation (LSI) 
Visualization of LSI. By LSI, our CTD can be easily trained with rectangular or quadrilater bounding boxes without extra manual efforts. Based on our recent research, the stronger supervision can also effectively improve the performance.

<div align=center><img src="images/in1.jpg" width="50%" ></div>

# Detecting Results 
<!-- <img src="images/table.png" width="100%"> -->
<img src="images/detect_results.png" width="100%">


# Labeling tool 
  For the labeling tool and specific details of the gound truths, please refer to data/README.md. 

# Citation
If you find our method or the dataset useful for your research, please cite 
```
@article{yuliang2017detecting,
  title={Detecting Curve Text in the Wild: New Dataset and New Solution},
  author={Yuliang, Liu and Lianwen, Jin and Shuaitao, Zhang and Sheng, Zhang},
  journal={arXiv preprint arXiv:1712.02170},
  year={2017}
}
```

# Requirement 
1. Clone this repository. ROOT is the directory where you clone.
2. cd ROOT/caffe/  and use your own Makefile.config to compile (make all && make pycaffe). If you are using ubuntu 14.04, you may need to modify Makefile line 181 (hdf5_serial_hl hdf5_serial) to (hdf5 hdf5_hl).
3. cd ROOT/lib make (based on python2)
4. pip install shapely. (Enable computing polygon intersection.)

# Installation

参考py-R-FCN

# Installation supporting cpu-only

参考py-RFCN, 比较文件差别再改

查看网络结构: 添加了lstm, 使用smoothL1OHEM

convert, lstm,reverse支持cpu,BLSTM是LSTM的子类, cpu支持似乎有问题, softmaxLoss, smoothL1, roipooling都不支持cpu

OHEM需要配合BoxAnnotatorOHEM, 详见`rfcn_end2end-model/train_agnostic_ohem.prototxt`

暂时先不用OHEM, 改用smoothL1, 

smoothL1需要4个输入, smoothL1OHEM需要三个, 根据对in/outside weight的分析, 对于RCNN, 可以投机取巧地使用out = in

softmax_loss_layer.c*相比rfcn, 多了对top数目大于1的处理, 由于我们用到的只有一个top, 故考虑将相关部分注释掉, 与RFCN保持一致

# RCNN系列使用注意

train:

与一般不同: max_iter在`rfcn*.sh`设置,snapshot在`solver.prototxt`

会显示训练和的平均速度

test: 

只在训练结束后进行一次测试, 而且是通过python程序调用检测评价代码实现的

会显示测试的速度

测试过程是一张一张进行的, 没有batch-size, 

~~通过在solver中添加test_net可以做到在训练过程中测试, 但由于该版本caffe不支持检测的评价, ap-vesion, 只能输出分类准确率~~–由于test是对一张张图片进行的, 所以不能实现

解决办法: 或者在python的训练代码中加入测试

### 快速训练,测试代码

```shell
experiments/scripts/*.sh
experiments/cfgs/*.yml
lib/fast_rcnn/config.py
solver.prototxt
```

修改使用数据集的图像数目: config.py NUM_IMAGES



## Feedback
Suggestions and opinions of this dataset (both positive and negative) are greatly welcome. Please contact the authors by sending email to
`liu.yuliang@mail.scut.edu.cn`.

## Copyright
The SCUT-CTW1500 database is free to the academic community for research purpose usage only.

For commercial purpose usage, please contact Dr. Lianwen Jin: [eelwjin@scut.edu.cn](eelwjin@scut.edu.cn)

Copyright 2017, Deep Learning and Vision Computing Lab, South China China University of Technology. [http://www.dlvc-lab.net](http://www.dlvc-lab.net)