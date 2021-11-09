'''
对数据进行重采样，把5000个点重采样为2048个
'''

import os
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
# npy='./data/yes/100.npy'
# array=np.load(npy)
# ecg=array[:,0]
# x_ecg=[i for i in range(ecg.shape[0])]
# ecg_resampled=signal.resample(ecg,2048)
# x_ecg_resampled=[i for i in range(ecg_resampled.shape[0])]
# plt.subplot(1,2,1)
# plt.plot(x_ecg,ecg)
# plt.subplot(1,2,2)
# plt.plot(x_ecg_resampled,ecg_resampled)
# plt.show()
boxes = np.load('C:\\Users\\HP\\Desktop\\ecg\\data\\no\\61.npy')
print(boxes)
np.savetxt('C:\\Users\\HP\\Desktop\\ecg\\61.txt', boxes, fmt='%s', newline='\n')
print("sucessfully")