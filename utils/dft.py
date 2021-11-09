import math
from math import pi
from math import sin
from math import cos
# para=10+10j
# print(abs(para))
# sum=0
# length=40
# for i in range(length):
#     basis=math.cos(math.pi*2*(2*i)/length)-math.sin(math.pi*2*(2*i)/length)*1j
#
#     sum+=(math.cos(math.pi*2*(2*i)/length+pi/6)*basis)
#     #print(sum)
#
# print(sum)
# print (math.atan(10/17.3))
# print (pi/6)
#DFT y=sin (2*pi*x)



sum=0
re=0
im=0
N=400
K=400
for k in range(K):
    for n in range(N):
        re+=cos(2*pi*n)*cos(2*pi*k*n/40)
        im+=cos(2*pi*n)*sin(2*pi*k*n/40)*1j*(-1)
        sum=sum+re+im
    print (' k= ',k,re,' ',im)
    #print (k,'__',sum,abs(sum))
    #sum = 0
    re=0
    im=0

    #计算实部与虚部