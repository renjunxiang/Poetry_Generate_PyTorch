import torch
import torch.nn as nn
import config


# 定义网络结构,6629个字
class NET_RNN(nn.Module):
    def __init__(self):
        super(NET_RNN, self).__init__()
        self.embedding = nn.Sequential(
            nn.Embedding(num_embeddings=6630, embedding_dim=300)
        )
        self.lstm = nn.Sequential(
            nn.LSTM(300, 300, num_layers=2, batch_first=True)
        )
        self.output = nn.Sequential(
            nn.Linear(300, 6630),
        )

    def forward(self, x):
        x = self.embedding(x)
        out, (_, _) = self.lstm(x)
        x = self.output(out)
        x = x.view(-1, x.size()[-1])
        return x


class NET_BERT(nn.Module):
    def __init__(self):
        super(NET_BERT, self).__init__()
        self.output = nn.Sequential(
            nn.Linear(768 * len(config.BERT_LAYERS.split(',')), 6630),
        )

    def forward(self, x):
        x = self.output(x)
        x = x.view(-1, x.size()[-1])
        return x


class NET_BERT_RNN(nn.Module):
    def __init__(self):
        super(NET_BERT_RNN, self).__init__()
        self.lstm = nn.Sequential(
            nn.LSTM(768 * len(config.BERT_LAYERS.split(',')), 300,
                    num_layers=2, batch_first=True)
        )
        self.output = nn.Sequential(
            nn.Linear(300, 6630),
        )

    def forward(self, x):
        out, (_, _) = self.lstm(x)
        x = self.output(out)
        x = x.view(-1, x.size()[-1])
        return x
