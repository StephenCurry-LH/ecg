import pywt, os, copy
import torch
from collections import Counter
import numpy as np
from scipy.signal import resample
from torch.utils.data import Dataset
from sklearn.preprocessing import scale
from scipy import signal
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


class Dataset_DIASTOLIC(Dataset):#dadaset 重采样，归一化
    def __init__(self, data_path,train_list,train_label):
        self.data_path=data_path
        self.train_label=train_label
        self.train_list=train_list

    def __len__(self):
        return len(self.train_list)

    def get_statis_result(self,array):#数据归一化
        channel,length = array.shape#shape代表维数，channel代表几个通道，length代表长度
        min = np.min(array,axis=1)#每行的最小值
        max = np.max(array,axis=1)#每行的最大值
        temp_array = (array-min.reshape(channel,1)) / (max-min).reshape(channel,1)#reshape将数组重新组织
        return temp_array

    def _resample(self,array):
        channel,length=array.shape
        resampled_array=[]
        for c in range(channel):
            temp=signal.resample(array[c,:],224)
            resampled_array.append(temp)
        resampled_array=np.array(resampled_array)
        return resampled_array

    def __getitem__(self,idx):
        patient_name=os.path.join(self.data_path,self.train_list[idx])
        all_npy=os.listdir(patient_name)

        npy_name = os.path.join(patient_name, all_npy[0])
            
        array = np.load(npy_name,allow_pickle=True)

        normalizated_array=self.get_statis_result(array)
        normalizated_array=self._resample(normalizated_array)
        return normalizated_array.transpose(1,0) ,float(self.train_label[idx])#转置





if __name__=='__main__':

    dir_path='./data/diastolic'
    all_data=os.listdir(dir_path)
    all_label = [0 if npy[:npy.index('_')]=='0' else 1 for npy in all_data]


    x_train,x_test,y_train,y_test=train_test_split(all_data,all_label,test_size=0.4,random_state=42)
    result=Counter(y_train)
    #print(result.most_common(3))

    params={'batch_size':8,'shuffle':True,'num_workers':16,'pin_memory':True}
    train_set=Dataset_DIASTOLIC('./data/diastolic',x_train,y_train)
    train_loader=DataLoader(train_set,**params)

    for batch_idx,(x,y) in enumerate(train_loader):
        print(batch_idx,x.shape,np.array(y))
