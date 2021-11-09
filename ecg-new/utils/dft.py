import math
from math import pi
from math import sin
from math import cos
from matplotlib import pyplot as  plt

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

plot_y=[]
N=40
K=40
sum=0
re=0
im=0
for k in range(K):      #频率
    for n in range(N):  #第几个点
        re=cos(2*pi*2*n/K+pi/4)*cos(2*pi*k*n/K)
        im=cos(2*pi*2*n/K+pi/4)*sin(2*pi*k*n/K)*(-1j)
        #basis=cos(2*pi*n*k/K)-sin(2*pi*n*k/K)*1j
        #basis=cos(2*pi*k*n/40)

        sum+=re
        sum+=im


    print ('X'+str(k),sum)
    plot_y.append(sum)
    sum=0

x=[i for i in range(len(plot_y))]
y=list(map(abs,plot_y))

plt.figure()
plt.scatter(x,y)
plt.savefig('40samples.png')



# sum=0
# re=0
# im=0
# N=400
# K=400
# for k in range(K):
#     for n in range(N):
#         re+=cos(2*pi*n)*cos(2*pi*k*n/40)
#         im+=cos(2*pi*n)*sin(2*pi*k*n/40)*1j*(-1)
#         sum=sum+re+im
#     print (' k= ',k,re,' ',im)
#     #print (k,'__',sum,abs(sum))
#     #sum = 0
#     re=0
#     im=0
#
#     #计算实部与虚部