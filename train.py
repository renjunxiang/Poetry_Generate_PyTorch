import os
import pickle
from ProcessData import ProcessData
from net import NET_RNN, NET_BERT, NET_BERT_RNN, DatasetRNN, DatasetBERT
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset
from flair.embeddings import BertEmbeddings
import config

# 超参数设置
EPOCH = config.EPOCH
BATCH_SIZE = config.BATCH_SIZE
LR = config.LR

# 其他设置
POOLING = config.POOLING
BERT_LAYERS = config.BERT_LAYERS
EMBEDDING = config.EMBEDDING
LOG_BATCH_NUM = config.LOG_BATCH_NUM

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(method='RNN'):
    if method not in ['RNN', 'BERT', 'BERT_RNN']:
        raise ValueError("method should be 'RNN','BERT' or 'BERT_RNN'")
    DIR = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists('%s/data/x_seq.pkl' % DIR):
        with open('%s/data/Tang_Poetry.pkl' % DIR, 'rb') as f:
            texts = pickle.load(f)
        with open('%s/data/x_seq.pkl' % DIR, 'rb') as f:
            x_seq = pickle.load(f)
        with open('%s/data/y_seq.pkl' % DIR, 'rb') as f:
            y_seq = pickle.load(f)
    else:
        print('Data has not been processed, start processing!')
        processdata = ProcessData()
        texts = processdata.load_data(len_min=40, len_max=200)
        data_process = processdata.text2seq(texts=texts, num_words=None, maxlen=40)
        x_seq = torch.LongTensor(data_process['x_seq'])
        y_seq = torch.LongTensor(data_process['y_seq'])
        with open('%s/data/x_seq.pkl' % DIR, 'wb') as f:
            pickle.dump(x_seq, f)
        with open('%s/data/y_seq.pkl' % DIR, 'wb') as f:
            pickle.dump(y_seq, f)
        with open('%s/data/tokenizer.pkl' % DIR, 'wb') as f:
            pickle.dump(data_process['tokenizer'], f)

    if method == 'RNN':
        # 定义训练批处理数据
        trainloader = torch.utils.data.DataLoader(
            dataset=DatasetRNN(x_seq[:-1000], y_seq[:-1000]),
            batch_size=BATCH_SIZE, shuffle=True)

        testloader = torch.utils.data.DataLoader(
            dataset=DatasetRNN(x_seq[-1000:], y_seq[-1000:]),
            batch_size=BATCH_SIZE, shuffle=True)
        Net = NET_RNN
    else:
        processdata = ProcessData()
        texts = processdata.reshape_seqs(texts, maxlen=40, data_type='str')[:, :-1]
        embedding = BertEmbeddings(bert_model=EMBEDDING,
                                   pooling_operation=POOLING,
                                   layers=BERT_LAYERS)

        print('finish loading BERT')

        trainloader = torch.utils.data.DataLoader(
            dataset=DatasetBERT(texts[:-1000], y_seq[:-1000], embedding),
            batch_size=BATCH_SIZE, shuffle=True)
        testloader = torch.utils.data.DataLoader(
            dataset=DatasetBERT(texts[-1000:], y_seq[-1000:], embedding),
            batch_size=BATCH_SIZE, shuffle=True)
        if method == 'BERT':
            Net = NET_BERT
        else:
            Net = NET_BERT_RNN

    # 定义损失函数loss function和优化方式
    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LR)

    for epoch in range(EPOCH):
        sum_loss = 0.0
        # 数据读取
        for i, data in enumerate(trainloader):
            x_seq_batch, y_seq_batch = data
            x_seq_batch, y_seq_batch = x_seq_batch.to(device), y_seq_batch.to(device)
            y_seq_batch = y_seq_batch.flatten()

            # 梯度清零
            optimizer.zero_grad()

            # forward + backward
            outputs = net(x_seq_batch)
            loss = criterion(outputs, y_seq_batch)
            loss.backward()
            optimizer.step()

            # 每训练LOG_BATCH_NUM个batch打印一次平均loss
            sum_loss += loss.item()
            if i % LOG_BATCH_NUM == LOG_BATCH_NUM - 1:
                print('[%d, %d] loss: %.03f'
                      % (epoch + 1, i + 1, sum_loss / LOG_BATCH_NUM))
                sum_loss = 0.0

        # 每跑完一次epoch测试一下准确率
        with torch.no_grad():
            correct = 0
            total = 0
            for data in testloader:
                x_seq_batch, y_seq_batch = data
                x_seq_batch, y_seq_batch = x_seq_batch.to(device), y_seq_batch.to(device)
                y_seq_batch = y_seq_batch.flatten()
                outputs = net(x_seq_batch)
                _, predicted = torch.max(outputs.data, 1)
                total += y_seq_batch.size(0)
                correct += (predicted == y_seq_batch).sum()
            print('第%d个epoch的识别准确率为：%d%%' % (epoch + 1, (100 * correct / total)))
        torch.save(net.state_dict(), '%s/%s_%03d.pth' % ('./model', method, epoch + 1))


if __name__ == '__main__':
    # train(method='RNN')
    # train(method='BERT')
    train(method='BERT_RNN')
