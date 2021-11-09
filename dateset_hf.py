'''
数据集设置
'''

import pywt, os, copy
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import scale
from scipy import signal
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
all_data=[]
all_label=[]
all_yes_data=os.listdir('./data1/yes')
all_no_data=os.listdir('./data1/no')
all_yes_label=[1]*len(all_yes_data)
all_no_label=[0]*len(all_no_data)

all_data.extend(all_yes_data)
all_data.extend(all_no_data)

all_label.extend(all_yes_label)
all_label.extend(all_no_label)

x_train,x_test,y_train,y_test=train_test_split(all_data,all_label,test_size=0.3,random_state=42)

class DatasetECG(Dataset):
    def __init__(self, data_path,train_list,train_label):
        self.data_path=data_path
        self.train_label=train_label
        self.train_list=train_list

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self,idx):
        npy_name=os.path.join(self.data_path,self.train_list[idx])
        array=np.load(npy_name)
        return array,self.train_label[idx]

# params={'batch_size':32,'shuffle':True,'num_workers':16,'pin_memory':True}
#
# train_set=DatasetECG('./data/all',x_train,y_train)
# train_loader=DataLoader(train_set,**params)
#
# # for batch_idx,(x,y) in enumerate(train_loader):
# #     print(batch_idx,x.shape,y)








#
# class ECGDataset(Dataset):
#     """
#     A generic data loader where the samples are arranged in this way:
#     dd = {'train': train, 'val': val, "idx2name": idx2name, 'file2idx': file2idx}
#     """
#
#     def __init__(self, data_path, train=True):
#         super(ECGDataset, self).__init__()
#         dd = torch.load(config.train_data)
#         self.train = train
#         self.data = dd['train'] if train else dd['val']
#         self.idx2name = dd['idx2name']
#         self.file2idx = dd['file2idx']
#         self.wc = 1. / np.log(dd['wc'])
#
#     def __getitem__(self, index):
#         fid = self.data[index]
#         file_path = os.path.join(config.train_dir, fid)
#         df = pd.read_csv(file_path, sep=' ').values
#         x = transform(df, self.train)
#         target = np.zeros(config.num_classes)
#         target[self.file2idx[fid]] = 1
#         target = torch.tensor(target, dtype=torch.float32)
#         return x, target
#
#     def __len__(self):
#         return len(self.data)


