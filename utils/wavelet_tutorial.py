# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pywt
import read_ecg
data=read_ecg.ecg_signal()
t=[x for x in range(data.shape[0])]

from matplotlib.font_manager import FontProperties


sampling_rate = 102400  #采样点个数
t = np.arange(0, 1.0, 1.0 / sampling_rate)
f1 = 100
f2 = 200
f3 = 300
data = np.piecewise(t, [t < 1, t < 0.8, t < 0.3],
                    [lambda t: np.sin(2 * np.pi * f1 * t), lambda t: np.sin(2 * np.pi * f2 * t),
                     lambda t: np.sin(2 * np.pi * f3 * t)])
wavename = 'cgau8'  #haar小波等
totalscal = 10240*3   #二分法采样次数
fc = pywt.central_frequency(wavename)
cparam = 2 * fc * totalscal
scales = cparam / np.arange(totalscal, 1, -1)
[cwtmatr, frequencies] = pywt.cwt(data, scales, wavename, 1.0 / sampling_rate)
plt.figure(figsize=(16, 9))
plt.subplot(211)
plt.plot(t, data)
plt.xlabel(u"time(s)")
plt.title(u"300Hz 200Hz 100Hz Time spectrum")
plt.subplot(212)
plt.contourf(t, frequencies, abs(cwtmatr)) #shape= 5000,,1023,1023*5000
plt.ylabel(u"freq(Hz)")
plt.ylim(0,500)
plt.xlabel(u"time(s)")
plt.subplots_adjust(hspace=0.4)
#plt.show()
plt.savefig('./wavelet.png')
print("exit")
