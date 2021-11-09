from numpy import array as matrix, arange,zeros,transpose,matmul,ones
from math import sqrt,cos,pi
import numpy as np
np.set_printoptions(suppress=True)

a=matrix([[52,55,61,66,70,61,64,73],
          [63,59,55,90,109,85,69,72],
          [62,59,68,113,144,104,66,73,],
          [63,58,71,122,154,106,70,69],
          [67,61,68,104,126,88,68,70],
          [79,65,60,70,77,68,58,75],
          [85,71,64,59,55,61,65,83],
          [87,79,69,68,65,76,78,94]
          ])
a=a-128  #减少偏移

DCT_mat=np.zeros((8,8),dtype=float)

def dct_freq(u,v):
    alpha_u=1/sqrt(2) if u==0 else 1
    alpha_v = 1/sqrt(2) if v == 0 else 1
    theta=0.25*alpha_u*alpha_v  #因子，构成正交矩阵
    sum=0
    for x in range(8):
        for y in range(8):
            basis=a[x][y]*cos((2*x+1)*u*pi/16)*cos((2*y+1)*v*pi/16)
            sum+=basis
    return theta*sum
#获得u，v频率对应DCT变换

for u in range(8):
    for v in range(8):
        DCT_mat[u][v]=round(dct_freq(u,v),2) #赋值

print(DCT_mat)



