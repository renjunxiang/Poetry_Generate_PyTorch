import os
import pickle
from net import Mynet
import torch
from torch import nn
import re
import numpy as np

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 配置变量
dir_model = './model/net_003.pth'
dir_tokenizer = './data/tokenizer.pkl'
maxlen = 40


def test():
    with open(dir_tokenizer, 'rb') as f:
        tokenizer = pickle.load(f)
    e_index = tokenizer.word_index['e']

    net = Mynet().to(device)
    checkpoint = torch.load(dir_model)
    net.load_state_dict(checkpoint)

    while True:
        print('\n请输入文本，在此基础上作诗。不输入则随机开始，quit离开！\n')
        text = input('输入：')
        if text == 'quit':
            break
        elif text == '':
            text = np.random.choice(list(tokenizer.index_word.values()))
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
            if len(text) >= maxlen:
                break
        text_list = re.findall(pattern='[^。？！]*[。？！]', string=text)
        print('创作完成：\n')
        for i in text_list:
            print(i)


if __name__ == '__main__':
    test()
