# Graduation Project
**My graduation project of implementing rPPG based heart rate estimation.**

## 目录
1. [主要工作 Contribution](#主要工作)
2. [环境配置 Environment](#环境配置)
3. [代码功能 Function](#代码功能)

## 主要工作
1. 通过人脸关键点检测实现人脸区域的对齐及剪裁，接着对人脸区域进行划分并提取出各小区域（ROI）中像素平均值随时间变化的一维信号，最后进行心率频率范围的带通滤波。通过以上操作初步减小人体运动，环境光，电子设备等噪声。  
2. 根据心率信号的频率特征分别设计了区域信号自适应加权网络以及心率信号映射网络。其中区域信号自适应加权网络会根据信号本身的特征自适应的加权5个ROI上的信号以达到特征融合，而心率信号映射网络则将加权后的组合信号映射到唯一的心率值。本文将两个网络级联后进行联合训练并比较了DNN, CNN, RNN三种结构的网络性能。  
3. 经过6个数值指标（Mean, Standard Deviation, MSE, MAE, MAPE, Pearson Correlation Coefficient）以及3种可视化图像（模型学习曲线，Bland-Altman图，散点回归图）的全方位分析给出了最佳级联网络结构，即CNN+LSTM，其在BEAN-HR数据集上达到了[-10,12]bpm的95%一致性界限。  
4. 通过视频压缩实验以及人脸关键点检测稳定性实验分析了系统在VIPL-HR数据集上失效的原因，并同时证明了视频的压缩编码格式不会对本文设计的系统产生影响，视频编码格式包括AVC1，MJPG，FMP4，EM2V，DIVX，WMV3以及PIM1这些常用的格式。  
5. 对神经网络进行了可视化分析，给出了自适应加权网络在高低心率段上的6个样例的加权结果，直观的显示了网络的训练成果。  

## 环境配置
### 系统环境
Windows10  
CUDA10.1+cudnn7.6.5  
Visual Studio 2017

### 普通库安装(直接pip install)
tensorflow-gpu==2.1.0  
numpy==1.19.0  
keras==2.1.5  
opencv-python==4.5.1  
scipy==1.4.1  
pandas==1.1.5  
Pillow==8.1.0  
matplotlib==3.3.3  
cmake==3.18.4

以上安装均可运行命令：  
```python
pip install [库]==[版本号] -i https://pypi.tuna.tsinghua.edu.cn/simple
```  
### face_recognition库以及dlib库安装 
face_recognition背后调用的就是dlib库，因此需要先安装dlib：  
```python
pip install dlib==19.21.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
``` 
此时dlib库默认不使用GPU运算，因此要想运算快速则需要重新编译成支持GPU运算的版本，进入dlib的安装目录后运行：  
```python
python setup.py install
```
编译时会自动检测电脑是否支持GPU，很方便，最后再安装face_recognition：  
```python
pip install face_recognition -i https://pypi.tuna.tsinghua.edu.cn/simple
```

最后进入[环境检测Notebook](Env_Test.ipynb)，检查环境是否配置一致  

## 代码功能
1. [Frames-Alignment](Frames_Alignment.py)文件中包括了对原始视频进行人脸对齐，人脸框切割，人脸区域划分，肤色分割以及区域信号提取的代码。
2. [Cache-Signal](Cache_Signal.py)文件中包括了对提取出的ROI信号进行缓存，信号滤波的代码。
3. [ROI-Signal-Visualization](ROI_Signal_Visualization.py)文件对信号进行可视化显示。
4. [Network](Network.py)文件中包括了本文提出的级联网络结构的定义。
5. [Training](Training-Demo.ipynb)文件则为深度学习网络训练的整个pipeline，包括数据集的读取，数据增强，预处理，网络的训练，结果的可视化分析（B-A图，回归图）等。
6. [Video-Codec](Video-Codec.ipynb)文件的功能是进行视频编码格式转换，实验视频压缩对此系统的影响。
7. [Learning-Curve](Learning-Curve.ipynb)则可以对训练时保存的历史记录进行可视化。
8. [Dlib-Stability-Check](Dlib-Stability-Check-Demo.ipynb)用于测试dlib人脸关键点检测算法的稳定性。
9. [HR-Weight-Network-Visualization](HR-Weight-Network-Visualization-Demo.ipynb)文件的作用时对自适应加权网络给出的5个权重输出进行可视化分析。
10. [Inference](Inference-Demo.ipynb)用于对自己的视频进行测试，即可以拍摄一段还有人脸的视频，利用此系统进行心率值估计。
