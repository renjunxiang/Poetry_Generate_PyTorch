from keras.preprocessing.text import Tokenizer
import numpy as np
import os
import pickle
from creat_data import creat_data

DIR = os.path.dirname(os.path.abspath(__file__))


class ProcessData():
    def __init__(self):
        self.num_words = None
        self.texts_seq = None
        self.x_seq = None
        self.y_seq = None
        self.tokenizer = None

    def load_data(self, len_min=0, len_max=200):
        """
        导入数据
        :param len_min: 最短长度
        :param len_max: 最长长度
        :return:
        """
        # 创建数据
        creat_data()
        with open(DIR + '/data/Tang_Poetry.pkl', mode='rb') as f:
            texts = pickle.load(f)
        texts = [i for i in texts if (len(i) >= len_min) and (len(i) <= len_max)]

        return texts

    def text2seq(self, texts=None, num_words=None, maxlen=40):
        """
        maxlen=40时，合计69867样本数
        :param texts: 诗歌文本
        :param num_words: 保留字数量
        :param maxlen: 样本句子长度
        :return:
        """
        tokenizer = Tokenizer(num_words=num_words, char_level=True)
        tokenizer.fit_on_texts(texts)
        self.tokenizer = tokenizer
        print('Finish tokenizer,words_num = %d' % (len(tokenizer.word_index)))

        # 转编码
        texts_seq = tokenizer.texts_to_sequences(texts=texts)
        print('Finish texts_to_sequences')

        texts_new = []
        for num, text_seq in enumerate(texts_seq):
            mod = len(text_seq) % maxlen
            text_seq += ([0] * (maxlen - mod))
            # text_seq = np.array(text_seq).reshape([-1, maxlen]).flatten().tolist()
            texts_new += text_seq
            if (num + 1) % 1000 == 0:
                print(num + 1)
        print('Finish reshape')

        texts_seq = np.array(texts_new).reshape([-1, maxlen])
        # self.texts_seq = texts_seq
        # self.x_seq = texts_seq[:, :-1]
        # self.y_seq = texts_seq[:, 1:]
        data_process = {'tokenizer': tokenizer,
                        'x_seq': texts_seq[:, :-1],
                        'y_seq': texts_seq[:, 1:]}

        return data_process


if __name__ == '__main__':
    pass
