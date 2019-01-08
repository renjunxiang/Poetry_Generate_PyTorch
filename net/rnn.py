import torch
import torch.nn as nn

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义网络结构,6629个字
class Mynet(nn.Module):
    def __init__(self):
        super(Mynet, self).__init__()
        self.embedding = nn.Sequential(
            nn.Embedding(num_embeddings=6630, embedding_dim=300)
        )
        self.lstm = nn.Sequential(
            nn.LSTM(300, 300, num_layers=2, batch_first=True)
        )
        self.output = nn.Sequential(
            nn.Linear(300, 6630),
        )

    # 定义前向传播过程，输入为x
    def forward(self, x):
        x = self.embedding(x)
        out, (_, _) = self.lstm(x)
        x = self.output(out)
        x = x.view(-1, x.size()[-1])
        return x
