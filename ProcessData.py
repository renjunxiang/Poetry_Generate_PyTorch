from keras.preprocessing.text import Tokenizer
import numpy as np
import re
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

    def reshape_seqs(self, seqs, maxlen=40, data_type='list'):
        seqs_new = []
        if data_type != 'list':
            for num, seq in enumerate(seqs):
                mod = len(seq) % maxlen
                seq += ('e' * (maxlen - mod))
                # seq = seq[:(len(seq) - mod)]
                seqs_new += re.findall('\S', seq)
                if (num + 1) % 10000 == 0:
                    print(num + 1)
        else:
            for num, seq in enumerate(seqs):
                mod = len(seq) % maxlen
                seq += ([0] * (maxlen - mod))
                seqs_new += seq
                if (num + 1) % 10000 == 0:
                    print(num + 1)
        print('Finish reshape')
        seqs_new = np.array(seqs_new).reshape([-1, maxlen])

        return seqs_new

    def text2seq(self, texts=None, num_words=None, maxlen=40):
        """
        maxlen=40时,合计69867样本数,6629个字
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
        texts_seq = self.reshape_seqs(texts_seq, maxlen=maxlen)

        data_process = {'tokenizer': tokenizer,
                        'x_seq': texts_seq[:, :-1],
                        'y_seq': texts_seq[:, 1:]}

        return data_process


if __name__ == '__main__':
    pass
