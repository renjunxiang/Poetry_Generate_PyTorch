import os
import pickle
import re


# 原始text数据转pkl
def creat_data():
    DIR = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists('%s/data/Tang_Poetry.pkl' % DIR):
        print('Data has been created!')
    else:
        print('Data has not been created, start creating data!')
        path = '%s/data/Tang_Poetry' % DIR
        files = os.listdir(path)

        end = 'e'
        texts = []
        for file in files:
            with open(path + '/' + file, mode='r') as f:
                text = f.readlines()
                if text:
                    text = text[0]
                    text = re.sub(pattern='[_（）《》 ]', repl='', string=text)
                    texts.append(text + end)
                else:
                    continue

        with open('%s/data/Tang_Poetry.pkl' % DIR, mode='wb') as f:
            pickle.dump(texts, f)

        print('Finish creating data!')


if __name__ == '__main__':
    creat_data()
