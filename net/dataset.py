import torch
from torch.utils.data import Dataset
import os
import pickle


# 定义数据读取方式
class MyDataset(Dataset):
    def __init__(self, x_seq,y_seq):
        self.x_seq = x_seq
        self.y_seq = y_seq

    def __getitem__(self, index):
        return self.x_seq[index], self.y_seq[index]

    def __len__(self):
        return len(self.x_seq)
