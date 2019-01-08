import os
import pickle
from ProcessData import ProcessData
from net import Mynet, MyDataset
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset

# 超参数设置
EPOCH = 5
BATCH_SIZE = 64
LR = 0.01

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    DIR = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists('%s/data/x_seq.pkl' % DIR):
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

    # 定义训练批处理数据
    trainloader = torch.utils.data.DataLoader(
        dataset=MyDataset(x_seq[:50000], y_seq[:50000]),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    testloader = torch.utils.data.DataLoader(
        dataset=MyDataset(x_seq[-1000:], y_seq[-1000:]),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # 定义损失函数loss function和优化方式
    net = Mynet().to(device)
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

            # 每训练100个batch打印一次平均loss
            sum_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %d] loss: %.03f'
                      % (epoch + 1, i + 1, sum_loss / 100))
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
        torch.save(net.state_dict(), '%s/net_%03d.pth' % ('./model', epoch + 1))


if __name__ == '__main__':
    train()
