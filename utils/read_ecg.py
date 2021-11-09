import numpy as np
from matplotlib import pyplot as  plt
import pywt
def ecg_signal(file='../data/hf_round1_train/train/2.txt'):
    with open(file, 'r') as f:
        singals = []
        for line in f.readlines():
            line = line.strip().split()
            singals.append(line[1])
    singals = [int(x) for x in singals[1:]]
    singals = np.array(singals)
    x = np.array([i for i in range(singals.shape[0])])
    return singals
# txt_path='../data/hf_round1_train/train/2.txt'
# with open(txt_path,'r') as f:
#     singals=[]
#     for line in f.readlines():
#         line=line.strip().split()
#         singals.append(line[1])
# singals=[int(x) for x in singals[1:]]
# singals=np.array(singals)
# x=np.array([i for i in range(singals.shape[0])])
# plt.plot(x,singals)
# plt.show()

# sampling_rate=1024
# wavename='cgau8'
# totalscale=256
# fc=pywt.central_frequency(wavename)
# cparam=2*fc*totalscale
# scales=cparam/np.arange(totalscale,1,-1)
# [cwtmatr,frequncies]=pywt.cwt(singals,scales,wavename,1.0/sapling_rate)
