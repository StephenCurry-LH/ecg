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
# import utils.resample
#from scipy import signal


#
# class DatasetECG(Dataset):
#     def __init__(self, data_path,train_list,train_label):
#         self.data_path=data_path
#         self.train_label=train_label
#         self.train_list=train_list
#
#     def __len__(self):
#         return len(self.train_list)
#
#     def __getitem__(self,idx):
#         npy_name=os.path.join(self.data_path,self.train_list[idx])
#         array=resample(np.load(npy_name))
#
#         mean_temp=np.mean(array,axis=0)
#         std_temp=np.std(array,axis=0)
#         if 0 in std_temp:
#             print(array)
#
#         return (array-mean_temp)/std_temp,int(self.train_label[idx])


class Dataset_DIASTOLIC(Dataset):
    def __init__(self, data_path,train_list,train_label):
        self.data_path=data_path
        self.train_label=train_label
        self.train_list=train_list

    def __len__(self):
        return len(self.train_list)

    def get_statis_result(self,array):
        channel,length=array.shape
        min=np.min(array,axis=1)
        max=np.max(array,axis=1)
        temp_array=(array-min.reshape(channel,1))/(max-min).reshape(channel,1)
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
        patient_name=os.path.join(self.data_path,self.train_list[idx])  #
        all_npy=os.listdir(patient_name)

        npy_name = os.path.join(patient_name, all_npy[0])
        #print(npy_name)
            
        array = np.load(npy_name,allow_pickle=True)
        #array = self._resample(array)
        normalizated_array=self.get_statis_result(array)
        normalizated_array=self._resample(normalizated_array)
        #return np.expand_dims((array-mean_temp)/std_temp,axis=1) ,float(self.train_label[idx])
        return normalizated_array.transpose(1,0) ,float(self.train_label[idx])





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
