import os
import pickle
from net import NET_RNN, NET_BERT, NET_BERT_RNN
import torch
from torch import nn
import re
import numpy as np
from flair.data import Sentence
from flair.embeddings import BertEmbeddings
import config
import argparse

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 配置变量
dir_tokenizer = './data/tokenizer.pkl'

# 其他设置
POOLING = config.POOLING
BERT_LAYERS = config.BERT_LAYERS
EMBEDDING = config.EMBEDDING
MODEL_PATH_RNN = config.MODEL_PATH_RNN
MODEL_PATH_BERT = config.MODEL_PATH_BERT
MODEL_PATH_BERT_RNN = config.MODEL_PATH_BERT_RNN

parser = argparse.ArgumentParser()
parser.add_argument('--method', default='RNN', help='default method is RNN')
parser.add_argument('--maxlen', default=96, help='default maxlen is 96')
opt = parser.parse_args()

def test(method='RNN'):
    if method not in ['RNN', 'BERT', 'BERT_RNN']:
        raise ValueError("method should be 'RNN','BERT' or 'BERT_RNN'")
    with open(dir_tokenizer, 'rb') as f:
        tokenizer = pickle.load(f)
    e_index = tokenizer.word_index['e']

    if method == 'RNN':
        net = NET_RNN().to(device)
        checkpoint = torch.load(MODEL_PATH_RNN)
        net.load_state_dict(checkpoint)
    else:
        if method == 'BERT':
            net = NET_BERT().to(device)
            checkpoint = torch.load(MODEL_PATH_BERT)
        else:
            net = NET_BERT_RNN().to(device)
            checkpoint = torch.load(MODEL_PATH_BERT_RNN)
        net.load_state_dict(checkpoint)
        embedding = BertEmbeddings(bert_model_or_path=EMBEDDING,
                                   pooling_operation=POOLING,
                                   layers=BERT_LAYERS)
    while True:
        print('\n请输入文本，在此基础上作诗。不输入则随机开始，quit离开！\n')
        text = input('输入：')
        if text == 'quit':
            break
        elif text == '':
            text = np.random.choice(list(tokenizer.index_word.values()))

        if method == 'RNN':
            while True:
                x_seq_batch = tokenizer.texts_to_sequences(texts=[text])
                x_seq_batch = torch.LongTensor(x_seq_batch).to(device)
                with torch.no_grad():
                    outputs = net(x_seq_batch)
                predicted = nn.Softmax(dim=0)(outputs.data.cpu()[-1])
                predicted = np.random.choice(np.arange(len(predicted)), p=predicted.numpy())
                if predicted not in [0, e_index]:
                    text += tokenizer.index_word[predicted]
                else:
                    break
                if len(text) >= opt.maxlen:
                    break
        else:
            while True:
                text_p = ' '.join(text)
                sentence = Sentence(text_p)
                embedding.embed(sentence)
                x_seq_batch = torch.Tensor([[token.embedding.numpy() for token in sentence]])
                x_seq_batch = torch.Tensor(x_seq_batch).to(device)
                with torch.no_grad():
                    outputs = net(x_seq_batch)
                predicted = nn.Softmax(dim=0)(outputs.data.cpu()[-1])
                predicted = np.random.choice(np.arange(len(predicted)), p=predicted.numpy())
                if predicted not in [0, e_index]:
                    text += tokenizer.index_word[predicted]
                else:
                    break
                if len(text) >= opt.maxlen:
                    break
        text_list = re.findall(pattern='[^。？！]*[。？！]', string=text)
        print('创作完成：\n')
        for i in text_list:
            print(i)
        # print('text:', text)


if __name__ == '__main__':
    test(method=opt.method)
