import pywt, os, copy
import torch
from collections import Counter
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import scale
from scipy import signal
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from utils.resample import *



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
        return resample(array),int(self.train_label[idx])

if __name__=='__main__':

    dir_path='./data/multi_class'
    all_data=os.listdir(dir_path)
    all_label=[npy[:npy.index('_')] for npy in all_data]


    x_train,x_test,y_train,y_test=train_test_split(all_data,all_label,test_size=0.3,random_state=42)
    result=Counter(y_train)
    print(result.most_common(3))

    params={'batch_size':32,'shuffle':True,'num_workers':16,'pin_memory':True}
    train_set=DatasetECG('./data/multi_class',x_train,y_train)
    train_loader=DataLoader(train_set,**params)

    for batch_idx,(x,y) in enumerate(train_loader):
        print(batch_idx,x.shape,np.array(y))








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