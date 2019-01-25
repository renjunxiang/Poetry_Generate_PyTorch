import torch
from torch.utils.data import Dataset
import os
import pickle
from flair.data import Sentence


# 定义数据读取方式
class DatasetRNN(Dataset):
    def __init__(self, x_seq, y_seq):
        self.x_seq = x_seq
        self.y_seq = y_seq

    def __getitem__(self, index):
        return self.x_seq[index], self.y_seq[index]

    def __len__(self):
        return len(self.x_seq)


class DatasetBERT(Dataset):
    def __init__(self, texts, y_seq,embedding):
        self.texts = texts
        self.embedding = embedding
        self.y_seq = y_seq

    def __getitem__(self, index):
        text = ' '.join(self.texts[index])
        sentence = Sentence(text)
        self.embedding.embed(sentence)
        x = torch.Tensor([token.embedding.numpy() for token in sentence])
        return x, self.y_seq[index]

    def __len__(self):
        return len(self.y_seq)
