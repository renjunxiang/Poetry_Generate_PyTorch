# BERT embedding
EMBEDDING = 'bert-base-chinese'  # 模型名称
BERT_LAYERS = '-1'  # 获取的层数,最后一层比官方推荐的后四层效果好很多
POOLING = 'mean'  # 词向量计算方式

# train
EPOCH = 5
BATCH_SIZE = 64
LR = 0.01  # 学习率
LOG_BATCH_NUM = 5  # 日志打印频率

# test
MODEL_PATH_RNN = './model/RNN_003.pth'  # rnn模型位置
MODEL_PATH_BERT = './model/BERT_002.pth'  # bert模型位置
MODEL_PATH_BERT_RNN = './model/BERT_RNN_003.pth'  # bert_rnn模型位置
