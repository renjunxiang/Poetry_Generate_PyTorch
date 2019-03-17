# Poetry_Generate_PyTorch
学习用PyTorch创作唐诗

[![](https://img.shields.io/badge/Python-3.6-blue.svg)](https://www.python.org/)
[![](https://img.shields.io/badge/torch-1.0.0-brightgreen.svg)](https://pypi.org/project/torch/1.0.0)
[![](https://img.shields.io/badge/keras-2.2.0-brightgreen.svg)](https://pypi.org/project/keras/2.2.0)
[![](https://img.shields.io/badge/numpy-1.16.2-brightgreen.svg)](https://pypi.org/project/numpy/1.16.2/)
[![](https://img.shields.io/badge/flair-0.4.1-brightgreen.svg)](https://pypi.org/project/flair/0.4.1/)

## **项目简介**
最近在学习PyTorch和BERT，当然用唐诗生成作为NLP新手任务再好不过了~之前写了TensorFlow和Keras版本的，代码写的比较烂，借此机会也提高下自己的工程能力。<br>

## **模块结构**
结构比较简单，包括数据、预处理方法、网络模型、训练代码和测试代码：<br>
* **数据**：在data文件夹中，解压Tang_Poetry.zip得到.txt文档，来源<https://github.com/todototry/AncientChinesePoemsDB>；<br>
* **预处理**：ProcessData.py，用于合并、清洗每个txt文档，文本转编码、填充切片<br>
* **网络结构**：在文件夹net中，mynet.py、dataset.py分别是网络结构和pytorch的dataset读取方式，包含了2层lstm、BERT和BERT+2层lstm三种方式。<br>
* **训练模型**：train.py，训练网络，'RNN', 'BERT', 'BERT_RNN'三种方式<br>
* **模型存储**：model文件夹，保存训练的模型<br>
* **生成唐诗**：test.py，'RNN', 'BERT', 'BERT_RNN'三种方式，生成诗歌<br>

## **其他说明**
* 按照传统的Embedding+LSTM+Softmax，效果还可以。<br>
* BERT是通过flair模块调用的，训练过程因为会逐个计算BERT输出，速度非常慢，内存足够的话建议先批量转为[样本数, 句子长度, BERT输出维度]的tensor。<br>
* 使用BERT发现效果非常差，学不到唐诗的结构。原因猜测是训练和推断都是从左向右，应当采用Transformer中Decoder部分的下三角矩阵Multi-headed attention，而BERT采用的是Transformer中Encoder部分。在推断过程中，由于不知道未生成部分的文本，应该采用Future blinding的方式，而不是生成一个字就全局重新计算权重。flair官方推荐的是用BERT后四层输出拼接成[batchsize, len, 786*4]的方式，实际使用发现只用最后一层的效果略好。

## **成果展示**
![](https://github.com/renjunxiang/Poetry_Generate_PyTorch/blob/master/picture/demo.png)<br>
![](https://github.com/renjunxiang/Poetry_Generate_PyTorch/blob/master/picture/demo2.png)<br>