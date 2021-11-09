import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
from math import pi,sin
#from read_ecg import *

def resample(original_array,sample_number=2048,plot=False):
    original_t=[i for i in range(len(original_array))]  #原始时间
    resampled_array=signal.resample(original_array,sample_number) #采样信号
    resampled_t=[i for i in range(sample_number)] #采样时间
    if plot:
        plt.figure(figsize=(50,10))
        plt.subplot(2, 1, 1)
        plt.plot(original_t, original_array)
        plt.subplot(2, 1, 2)
        plt.plot(resampled_t, resampled_array)
        plt.savefig('./resample.png')

    return resampled_array

if __name__=='__main__':
    #x=np.arange(1,100,5)
    x=ecg_signal()
    original_array=np.sin(x)
    resampd_array=resample(original_array=original_array,sample_number=2048,plot=True)


