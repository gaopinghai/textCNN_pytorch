# -*- coding: utf-8 -*-
'''
从原数据中选取部分数据；
选取数据的title前两个字符在字典WantedClass中；
且各个类别的数量为WantedNum
'''
import jieba
import json

TrainJsonFile = 'baike_qa2019/baike_qa_train.json'
ValidJsonFile = 'baike_qa2019/baike_qa_valid.json'
MyTainJsonFile = 'baike_qa2019/my_traindata.json'
MyValidJsonFile = 'baike_qa2019/my_validdata.json'
StopWordFile = 'stopword.txt'

WantedClass = {'教育': 0, '健康': 0, '生活': 0, '娱乐': 0, '游戏': 0}
WantedNum = 1000
numWantedAll = WantedNum * 5


def main():
    Datas = open(ValidJsonFile, 'r', encoding='utf_8').readlines()
    f = open(MyValidJsonFile, 'w', encoding='utf_8')

    numInWanted = 0
    for line in Datas:
        data = json.loads(line)
        cla = data['category'][0:2]
        if cla in WantedClass and WantedClass[cla] < WantedNum:
            json_data = json.dumps(data, ensure_ascii=False)
            f.write(json_data)
            f.write('\n')
            WantedClass[cla] += 1
            numInWanted += 1
            if numInWanted >= numWantedAll:
                break


if __name__ == '__main__':
    main()