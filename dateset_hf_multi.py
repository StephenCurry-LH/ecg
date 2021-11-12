import pywt, os, copy
import torch
from collections import Counter
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import scale
from scipy import signal
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
def maxminnorm(array):
    maxcols = array.max(axis = 0)
    mincols = array.min(axis = 0)
    data_shape = array.shape#返回的数组的维数
    data_rows = data_shape[0]
    data_cols = data_shape[1]
    t = np.empty((data_rows,data_cols))#返回一个随机数组
    for i in range(data_cols):
        t[:,i] = (array[:,i] - mincols[i]) / (maxcols[i] - mincols[i])
    return t

class DatasetECG(Dataset):
    def __init__(self, data_path,train_list,train_label):
        self.data_path=data_path
        self.train_label=train_label
        self.train_list=train_list

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self,idx):
        npy_name=os.path.join(self.data_path,self.train_list[idx])  #0npy,1npy,2npy,allnpy,0png,1png
        npy_name=sorted(npy_name)
        array=np.load(npy_name[0])
        array = maxminnorm(array)
        #a = torch.nn.Softmax()(array)
        #ecg_array = maxminnorm(array)
        # return ecg_array,int(self.train_label[idx])
        #max=np.max(array,axis=1)
        #min=np.min(array,axis=1)

        return array, int(self.train_label[idx])
    #返回的是原数据和标签

if __name__=='__main__':
    dir_path='./data/multi_class'
    all_data=os.listdir(dir_path)
    all_label=[npy[:npy.index('_')] for npy in all_data]

    x_train,x_test,y_train,y_test=train_test_split(all_data,all_label,test_size=0.3,random_state=42)
    result=Counter(y_train)
    print(result.most_common(3))

    params={'batch_size':1,'shuffle':True,'num_workers':16,'pin_memory':True}
    train_set=DatasetECG('./data/multi_class',x_train,y_train)
    # for list in train_set :
    #     print(list[0])
    #     print(list[1])
    train_loader=DataLoader(train_set,**params)

    for batch_idx,(x,y) in enumerate(train_loader):#x代表array，y代表标签
        print(batch_idx,x.shape,np.array(y),y.size(0))#batch_idx代表数组下标，（x,y）整个代表数据，是一个mini-batch,x是32*512*4的数组
        #y是32*1的数组，因为mini-batch = 32

