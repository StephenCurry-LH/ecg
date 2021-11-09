import matplotlib.pyplot as plt
import numpy as np
import pywt
import read_ecg
data=read_ecg.ecg_signal()
t=[x for x in range(data.shape[0])]
from matplotlib.font_manager import FontProperties
sampling_rate = 5000 #采样点个数
t = np.arange(0, 1.0, 1.0 / sampling_rate)
wavename = 'cgau8'  #haar小波等
totalscal = 1024   #二分法采样次数
fc = pywt.central_frequency(wavename)
cparam = 2 * fc * totalscal
scales = cparam / np.arange(totalscal, 1, -1)
[cwtmatr, frequencies] = pywt.cwt(data, scales, wavename, 1.0 / sampling_rate)
plt.figure(figsize=(16, 9))
plt.subplot(211)
plt.plot(t, data)
plt.xlabel(u"time(s)")
plt.title(u"Time spectrum")
plt.subplot(212)
plt.contourf(t, frequencies, abs(cwtmatr)) #shape= 5000,,1023,1023*5000
plt.ylabel(u"freq(Hz)")
plt.ylim(0,500)
plt.xlabel(u"time(s)")
plt.subplots_adjust(hspace=0.4)
#plt.show()
plt.savefig('./wavelet.png')
print("exit")
