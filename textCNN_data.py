from torch.utils.data import Dataset, DataLoader
import torch
import random
import numpy as np


trainDataFile = 'traindata_vec.txt'
valDataFile = 'valdata_vec.txt'


def get_valdata(file=valDataFile):
    valData = open(valDataFile, 'r').read().split('\n')
    valData = list(filter(None, valData))
    random.shuffle(valData)

    return valData


class textCNN_data(Dataset):
    def __init__(self):
        trainData = open(trainDataFile, 'r').read().split('\n')
        trainData = list(filter(None, trainData))
        random.shuffle(trainData)
        self.trainData = trainData

    def __len__(self):
        return len(self.trainData)

    def __getitem__(self, idx):
        data = self.trainData[idx]
        data = list(filter(None, data.split(',')))
        data = [int(x) for x in data]
        cla = data[0]
        sentence = np.array(data[1:])

        return cla, sentence



def textCNN_dataLoader(param):
    dataset = textCNN_data()
    batch_size = param['batch_size']
    shuffle = param['shuffle']
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    dataset = textCNN_data()
    cla, sen = dataset.__getitem__(0)

    print(cla)
    print(sen)