# MobileNetV2
### 选择语言 | Language
[中文简介](#简介) | [English](#Introduction)

### 结果 | Result
<img width="2480" height="1914" alt="mobilenetv2_training_curve" src="https://github.com/user-attachments/assets/85327b01-bd77-40dd-97e0-f4044e59c599" />


---

## 简介
MobileNetV2 是由谷歌团队于 2018 年在 MobileNetV1 基础上迭代提出的**新一代轻量化卷积神经网络**，相关成果发表于《MobileNetV2: Inverted Residuals and Linear Bottlenecks》。针对 MobileNetV1 低维特征经过 ReLU 激活易丢失信息、深层网络梯度退化、特征表达能力不足等痛点，MobileNetV2 引入两大核心创新：**倒残差结构（Inverted Residual）** 与 **线性瓶颈（Linear Bottleneck）**，同时沿用并优化宽度乘数、引入通道规整化策略，在更低计算量下实现远超 V1 的特征提取能力与分类精度。网络全程采用 ReLU6 适配移动端量化部署，通过 `make_divisible` 将通道数规整为 8 的倍数，贴合硬件并行计算特性。该模型在保持轻量化、低参数量、低算力消耗的同时，大幅缓解低维特征信息损耗，成为移动端分类、检测、分割、嵌入式视觉任务的骨干基准网络，后续几乎所有轻量网络均借鉴其倒残差与线性瓶颈设计思想。


## 架构
MobileNetV2 整体为**初始卷积 + 倒残差块堆叠 + 全局池化分类**的端到端轻量化网络，整体分为「基础卷积初始模块」「倒残差瓶颈特征提取模块」和「全局池化+分类头输出模块」三大核心部分，原论文标准输入为224×224分辨率的3通道RGB图像，适配通用图像分类任务，具体结构与核心设计如下：
- **基础初始模块**：网络首层采用标准3×3普通卷积，搭配BN批量归一化与ReLU6激活函数，完成浅层基础特征提取与初步下采样，为后续倒残差块提供高维特征输入。
- **轻量化特征提取模块（核心）**：以**倒残差瓶颈块**为基本单元堆叠构建，遵循「升维扩展→深度可分离卷积特征提取→降维线性压缩」的设计逻辑；先通过1×1卷积扩展通道提升特征丰富度，再利用3×3深度卷积做空间特征建模，最后用1×1卷积压缩通道且**不使用激活函数**构成线性瓶颈，避免低维特征被ReLU破坏。当步长为1且输入输出通道一致时引入残差捷径连接，增强梯度回流与特征复用能力；严格按照论文标准配置重复堆叠多组倒残差块，交替步长实现多尺度下采样。
- **分类输出模块**：后端采用自适应全局平均池化将特征图压缩为1×1，经特征展平后接入分类头；分类头由Dropout正则化层+单层全连接层组成，Dropout抑制过拟合，全连接层映射至对应类别维度，结构简洁高效、参数量极小。

该架构在继承MobileNetV1深度可分离卷积轻量化优势的同时，通过倒残差与线性瓶颈解决低维特征信息丢失问题，结合通道规整化、ReLU6量化友好设计，兼顾**速度、体积、精度、硬件部署**四大优势，是工业界移动端与边缘设备最常用的轻量级骨干网络之一。

<img width="705" height="161" alt="image" src="https://github.com/user-attachments/assets/2e98d6a2-007f-4c6a-a998-8c5e890e7262" />

<img width="831" height="370" alt="image" src="https://github.com/user-attachments/assets/19b3f095-2415-43a1-aff1-418ace961d5e" />

为什么需要升维：
<img width="1045" height="203" alt="image" src="https://github.com/user-attachments/assets/f8a93be7-8130-4303-a8a8-5875ee8d0247" />
我们选用的维度不够时，经过 ReLU 运算后，原本的分布就会被破坏，有一部分的信息将永远丢失，只有高纬度的时候才会相对保留。

倒残差模块：
<img width="741" height="366" alt="image" src="https://github.com/user-attachments/assets/3bbff933-4c3c-47c6-bede-76455d296740" />
<img width="789" height="645" alt="image" src="https://github.com/user-attachments/assets/a55510fb-d9c6-4640-91a8-8547e5460cb8" />
<img width="1069" height="789" alt="image" src="https://github.com/user-attachments/assets/dce1df1b-3473-44f2-a028-8222efa5fe96" />

网络架构：
<img width="1554" height="759" alt="image" src="https://github.com/user-attachments/assets/08e65e28-dda1-41f8-9b47-ac4cbca52d35" />

**注意**：我们使用的是数据集CIFAR-10，它是10类数据，并且不同于原文献，由于 CIFAR-10 图像尺寸（32×32）远小于原论文的 224×224，我们会对网络结构做微小适配（主要调整下采样步长、防止特征图尺寸过早压缩至消失），但核心架构**倒残差瓶颈块堆叠、线性无ReLU瓶颈、宽度乘数缩放、通道8倍数规整、ReLU6激活+残差连接**完全保留，严格复现原版MobileNetV2核心设计思想。

## 数据集
我们使用的是数据集CIFAR-10，是一个更接近普适物体的彩色图像数据集。CIFAR-10 是由Hinton 的学生Alex Krizhevsky 和Ilya Sutskever 整理的一个用于识别普适物体的小型数据集。一共包含10 个类别的RGB 彩色图片：飞机（ airplane ）、汽车（ automobile ）、鸟类（ bird ）、猫（ cat ）、鹿（ deer ）、狗（ dog ）、蛙类（ frog ）、马（ horse ）、船（ ship ）和卡车（ truck ）。每个图片的尺寸为32 × 32 ，每个类别有6000个图像，数据集中一共有50000 张训练图片和10000 张测试图片。
数据集链接为：https://www.cs.toronto.edu/~kriz/cifar.html

它不同于我们常见的图片存储格式，而是用二进制优化了储存，当然我们也可以将其复刻出来为PNG等图片格式，但那会很大，我们的目标是神经网络，这里不做细致解析数据集。

---

## Introduction
MobileNetV2 is a new generation lightweight convolutional neural network proposed by Google in 2018 on the basis of MobileNetV1, published in the paper *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. Aiming at the shortcomings of MobileNetV1, such as easy loss of low-dimensional feature information after ReLU activation, gradient degradation of deep networks and insufficient feature expression ability, MobileNetV2 introduces two core innovations: **Inverted Residual** and **Linear Bottleneck**. It inherits and optimizes the width multiplier, and adds a channel normalization strategy to round channels to multiples of 8 for hardware friendliness. Adopting ReLU6 activation throughout the network adapts to mobile terminal quantization deployment. It achieves far stronger feature extraction ability and classification accuracy than V1 with lower computational cost. It maintains the advantages of lightweight, small parameters and low computing consumption, and greatly alleviates the loss of low-dimensional feature information. It has become a benchmark backbone network for mobile classification, detection, segmentation and embedded vision tasks, and almost all subsequent lightweight networks learn from its inverted residual and linear bottleneck design.

## Architecture
The overall structure of MobileNetV2 consists of three core parts: initial convolution module, inverted residual bottleneck feature extraction module, and global pooling & classifier head output module. The original paper takes 224×224 RGB images as standard input for general image classification tasks.
- **Initial Basic Module**: The first layer uses standard 3×3 convolution with BN and ReLU6 activation, completing shallow feature extraction and initial downsampling to provide high-dimensional feature input for subsequent blocks.
- **Lightweight Feature Extraction Module (Core)**: Built by stacking inverted residual bottleneck blocks, following the logic of **channel expansion → depthwise convolution feature extraction → linear channel compression**. The 1×1 convolution first expands channels to enrich features, 3×3 depthwise convolution models spatial information, and the last 1×1 convolution compresses channels without activation to form a linear bottleneck, preventing low-dimensional feature damage by ReLU. Shortcut residual connection is added when stride=1 and input/output channels are consistent to enhance gradient backpropagation and feature reuse.
- **Classification Output Module**: Adaptive global average pooling compresses feature maps to 1×1, then flattens features and sends them to the classifier head. The classifier head consists of Dropout for regularization and a fully connected layer for category mapping, which effectively suppresses overfitting and keeps the network lightweight.

<img width="705" height="161" alt="image" src="https://github.com/user-attachments/assets/2e98d6a2-007f-4c6a-a998-8c5e890e7262" />

<img width="831" height="370" alt="image" src="https://github.com/user-attachments/assets/19b3f095-2415-43a1-aff1-418ace961d5e" />

Why is dimensionality increase necessary?

<img width="994" height="150" alt="image" src="https://github.com/user-attachments/assets/364151ec-dbdb-4f65-80d1-b3324e076b93" />

When the chosen dimension is insufficient, the original distribution will be destroyed after ReLU operation, and some information will be lost forever. Only at higher dimensions will it be relatively preserved.

Inverted residual module:
<img width="741" height="366" alt="image" src="https://github.com/user-attachments/assets/3bbff933-4c3c-47c6-bede-76455d296740" />
<img width="789" height="645" alt="image" src="https://github.com/user-attachments/assets/a55510fb-d9c6-4640-91a8-8547e5460cb8" />
<img width="1069" height="789" alt="image" src="https://github.com/user-attachments/assets/dce1df1b-3473-44f2-a028-8222efa5fe96" />

Network architecture:
<img width="1554" height="759" alt="image" src="https://github.com/user-attachments/assets/08e65e28-dda1-41f8-9b47-ac4cbca52d35" />


**Note:** We use the CIFAR-10 dataset with 10 classification categories. Since the 32×32 image size of CIFAR-10 is much smaller than the 224×224 input in the original paper, slight adjustments are made to the downsampling stride to avoid excessive feature compression. However, the core design of **inverted residual bottleneck, linear bottleneck without ReLU, width multiplier, channel rounding to multiples of 8 and residual connection** is completely consistent with the original MobileNetV2.

## Dataset
We used the CIFAR-10 dataset, a color image dataset that more closely approximates common objects. CIFAR-10 is a small dataset for recognizing common objects, compiled by Alex Krizhevsky and Ilya Sutskever. It contains RGB color images for 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. Each image is 32 × 32 pixels, with 6000 images per category. The dataset contains 50,000 training images and 10,000 test images.

The dataset link is: https://www.cs.toronto.edu/~kriz/cifar.html

---
## 原文章 | Original article
Sandler M, Howard A, Zhu M, et al. MobileNetV2: Inverted Residuals and Linear Bottlenecks[EB/OL]. arXiv preprint arXiv:1801.04381, 2018.
