# Poetry_Generate_PyTorch
学习用PyTorch创作唐诗

[![](https://img.shields.io/badge/Python-3.6-blue.svg)](https://www.python.org/)
[![](https://img.shields.io/badge/torch-0.4.1-brightgreen.svg)](https://pypi.python.org/pypi/torch/0.4.1)
[![](https://img.shields.io/badge/keras-2.2.0-brightgreen.svg)](https://pypi.python.org/pypi/keras/2.2.0)
[![](https://img.shields.io/badge/numpy-1.14.3-brightgreen.svg)](https://pypi.python.org/pypi/numpy/1.15.3)

## **项目简介**
最近在学习PyTorch，当然用唐诗生成作为NLP新手任务再好不过了~之前写了TensorFlow和Keras版本的，代码写的比较烂，借此机会也提高下自己的工程能力。<br>

## **模块结构**
结构比较简单，包括数据、预处理方法、网络模型、训练代码和测试代码：<br>
* **数据**：在data文件夹中，解压Tang_Poetry.zip，全唐诗txt文档，来源<https://github.com/todototry/AncientChinesePoemsDB>；<br>
* **预处理**：ProcessData.py，用于合并、清洗每个txt文档，文本转编码、填充切片<br>
* **网络**：在文件夹net中，rnn.py、dataset.py分别是2层lstm的网络和pytorch的dataset读取方式（不太习惯数据读取还要单独写一个方法）<br>
* **训练**：train.py，训练网络<br>
* **模型**：model文件夹，保存训练的模型<br>
* **生成**：test.py，生成诗歌<br>

## **成果展示**
![](https://github.com/renjunxiang/Poetry_Generate_PyTorch/blob/master/picture/demo.png)<br>