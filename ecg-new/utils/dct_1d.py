import numpy as np
from math import sqrt,cos,pi

x=[0.0,0.0,2.0,2.0,2.0,2.0,0.0,0.0]

def dct_freq(u):
    if u==0:
        alpha_u=sqrt(1/8)
    else:
        alpha_u=sqrt(2/8)
    sum=0
    for i in range(8):
        temp=x[i] * cos((i + 0.5) * pi / 8 * u)
        sum+=temp
    return sum*alpha_u

for i in range(8):
    f=dct_freq(i)
    print(round(f,2))