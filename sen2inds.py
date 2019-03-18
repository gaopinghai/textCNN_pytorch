#-*- coding: utf_8 -*-

import json
import sys, io
import jieba
import random

sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='gb18030') #改变标准输出的默认编码

trainFile = 'baike_qa2019/my_traindata.json'
stopwordFile = 'stopword.txt'
wordLabelFile = 'wordLabel.txt'
trainDataVecFile = 'traindata_vec.txt'
maxLen = 20

labelFile = 'label.txt'
def read_labelFile(file):
    data = open(file, 'r', encoding='utf_8').read().split('\n')
    label_w2n = {}
    label_n2w = {}
    for line in data:
        line = line.split(' ')
        name_w = line[0]
        name_n = int(line[1])
        label_w2n[name_w] = name_n
        label_n2w[name_n] = name_w

    return label_w2n, label_n2w


def read_stopword(file):
    data = open(file, 'r', encoding='utf_8').read().split('\n')

    return data


def get_worddict(file):
    datas = open(file, 'r', encoding='utf_8').read().split('\n')
    datas = list(filter(None, datas))
    word2ind = {}
    for line in datas:
        line = line.split(' ')
        word2ind[line[0]] = int(line[1])
    
    ind2word = {word2ind[w]:w for w in word2ind}
    return word2ind, ind2word


def json2txt():
    label_dict, label_n2w = read_labelFile(labelFile)
    word2ind, ind2word = get_worddict(wordLabelFile)

    traindataTxt = open(trainDataVecFile, 'w')
    stoplist = read_stopword(stopwordFile)
    datas = open(trainFile, 'r', encoding='utf_8').read().split('\n')
    datas = list(filter(None, datas))
    random.shuffle(datas)
    for line in datas:
        line = json.loads(line)
        title = line['title']
        cla = line['category'][0:2]
        cla_ind = label_dict[cla]

        title_seg = jieba.cut(title, cut_all=False)
        title_ind = [cla_ind]
        for w in title_seg:
            if w in stoplist:
                continue
            title_ind.append(word2ind[w])
        length = len(title_ind)
        if length > maxLen + 1:
            title_ind = title_ind[0:21]
        if length < maxLen + 1:
            title_ind.extend([0] * (maxLen - length + 1))
        for n in title_ind:
            traindataTxt.write(str(n) + ',')
        traindataTxt.write('\n')


def main():
    json2txt()


if __name__ == "__main__":
    main()