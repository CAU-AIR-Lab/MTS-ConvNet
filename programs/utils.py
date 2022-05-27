import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def load_config(dataset):
    if dataset == "WISDM":
        epoches = 500
        input_nc = 3
        segment_size = 60
        class_num = 6 
    elif dataset == "UCI_HAR":
        epoches = 500
        input_nc = 6
        segment_size = 128
        class_num = 6
    elif dataset == "OPPORTUNITY":
        epoches = 500
        input_nc = 113
        segment_size = 150
        class_num = 18 
    elif dataset == "PAMAP2":
        epoches = 500
        input_nc = 31
        segment_size = 512
        class_num = 18
    return epoches, input_nc, segment_size, class_num

def Read_Data(dataset, input_nc):
    data_path = os.path.join('Data', 'preprocessed', dataset)
    train_X = np.load(data_path+'/train_x.npy')
    train_Y = np.load(data_path+'/train_y.npy')
    test_X = np.load(data_path+'/test_x.npy')
    test_Y = np.load(data_path+'/test_y.npy')

    return To_DataSet(train_X, train_Y), To_DataSet(test_X, test_Y), test_Y

class To_DataSet(Dataset):
    def __init__(self, X, Y):
        self.data_num = Y.shape[0]
        self.x = torch.as_tensor(X)
        self.y = torch.as_tensor(Y)#torch.max(torch.as_tensor(Y), 1)[1]
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    def __len__(self):
        return self.data_num



def input_pipeline(dataset, input_nc, bs):
    train_data, eval_data, y_test_unary  = Read_Data(dataset, input_nc)
    train_queue = DataLoader(
        train_data, batch_size=bs,shuffle=True,
        pin_memory=True, num_workers=0)
    eval_queue = DataLoader(
        eval_data, batch_size=bs,shuffle=False,
        pin_memory=True, num_workers=0)
    return train_queue, eval_queue, y_test_unary

def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

def accuracy(output, target):
    _, predicted = torch.max(output.data, 1)
    total = target.size(0)
    correct = (predicted == target).sum()

    return float(correct) / total
