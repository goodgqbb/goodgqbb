import numpy as np
import json
import paddle
import paddle as P
import paddle.nn as nn
import paddle.nn.functional as F
from datetime import datetime
from flask import render_template, request,Flask
from run import app
import matplotlib.pyplot as plt
import  os
import jieba
L=125
dataset = json.load(open(r'/app/唐诗.json', encoding='UTF-8'))

def get_allchars(dataset):
    allchars = []
    for i in range(len(dataset)):
        allchars += [dataset[i][4].replace('\n', '')]
    return ''.join(allchars)

allchars = get_allchars(dataset)


def get_dict(allchars):
    char_freq_dict = dict()
    for char in allchars:
        if char not in char_freq_dict:
            char_freq_dict[char] = 0
        char_freq_dict[char] += 1
    char_freq_dict = sorted(char_freq_dict.items(), key = lambda x:x[1], reverse = True)
    char2id_dict = dict()
    id2char_dict = dict()
    n = 0
    for i in range(len(char_freq_dict)):
        char2id_dict[char_freq_dict[i][0]] = i
        id2char_dict[i] = char_freq_dict[i][0]
    return char2id_dict, id2char_dict

char2id_dict, id2char_dict = get_dict(allchars)
print(len(char2id_dict))
char2id_dict['</s>'] = 7394
char2id_dict['<START>'] = 7395
char2id_dict['<END>'] = 7396
id2char_dict[7394] = '</s>'
id2char_dict[7395] = '<START>'
id2char_dict[7396] = '<END>'



class Net(paddle.nn.Layer):
    def __init__(self, vocab_size=len(char2id_dict), embedding_dim=256, hidden_dim=512, num_layers=3):
        super(Net, self).__init__()
        self.embeddings = paddle.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = paddle.nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
        )
        self.linear = paddle.nn.Linear(in_features=hidden_dim, out_features=vocab_size)

    def forward(self, x, hidden=None):
        y = self.embeddings(x)
        if hidden is None:
            y, hidden = self.lstm(y)
        else:
            y, hidden = self.lstm(y, hidden)
        y = self.linear(y)
        return y, hidden


def show1(result, prefix=None, net=None, char2id_dict=char2id_dict, id2char_dict=id2char_dict, L=L):
    if net == None:
        net = Net()
    net.eval()
    result = list(result)
    len_result = len(result)
    x = P.to_tensor([char2id_dict['<START>']]).reshape([1, 1])
    hidden = None

    if prefix:
        for i in range(len(prefix)):
            _, hidden = net(x, hidden)
            x = P.to_tensor([char2id_dict[prefix[i]]]).reshape([1, 1])

    for i in range(len_result):
        _, hidden = net(x, hidden)
        x = P.to_tensor([char2id_dict[result[i]]]).reshape([1, 1])

    for i in range(len_result, L):
        x, hidden = net(x, hidden)
        _, top_index = P.topk(x, 1)
        # result += id2char_dict[top_index.numpy()[0][0][0]]
        x = top_index.reshape([1, 1])
        top_index = top_index.numpy()[0][0][0]
        if id2char_dict[top_index] == '<END>':
            # del result[-1]
            break
        result += id2char_dict[top_index]

    return ''.join(result)


def show2(acrostic, prefix=None, net=None, char2id_dict=char2id_dict, id2char_dict=id2char_dict):
    if net == None:
        net = Net()
    net.eval()
    len_acrostic = len(acrostic)
    x = P.to_tensor([char2id_dict['<START>']]).reshape([1, 1])
    hidden = None
    result = []

    if prefix:
        for i in range(len(prefix)):
            _, hidden = net(x, hidden)
            x = P.to_tensor([char2id_dict[prefix[i]]]).reshape([1, 1])

    previous = '<START>'
    n = 0  # 已完成的藏头
    for i in range(L):
        # previous = id2char_dict[x.numpy()[0][0]]
        x, hidden = net(x, hidden)
        _, top_index = P.topk(x, 1)

        if previous in {'。', '，', '?', '！', '<START>'}:
            if n == len(acrostic):
                break
            x = P.to_tensor([char2id_dict[acrostic[n]]]).reshape([1, 1])
            n += 1
        else:
            x = top_index.reshape([1, 1])
        previous = id2char_dict[x.numpy()[0][0]]
        result += previous

    return ''.join(result)


net = Net()
model_path = r'/app/net.pdparams'
net.set_state_dict(P.load(model_path))



changtoushis = ""
xuxieshi1 = ""

@app.route('/upload', methods=['POST'])
def upload():
#     all_files = [f for f in os.listdir('/app')]
#     return str(all_files) #获取当前工作目录路径
    global changtoushis
    text = request.form.get('text')
    return text
    acrostic = json.loads(text)
    print(acrostic)
    changtoushis = show2(acrostic=acrostic, prefix='一二三四五六七，七六五四三二一。', net = net)
    print(changtoushis)
    return changtoushis

@app.route('/upload1', methods=['POST'])
def upload1():
    global xuxieshi1
    text = request.form.get('text')
    print(text)
    acrostic = json.loads(text)
    print(acrostic)
    xuxieshi1 = show1(result=acrostic, prefix='明日何其多，总管一二三。', net = net)
    print(xuxieshi1)
    return xuxieshi1

