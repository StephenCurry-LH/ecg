import numpy as np
from matplotlib import pyplot as plt
import cv2
x = np.arange(-10,10,0.1)
y = np.arange(-10,10,0.1)
xx,yy = np.meshgrid(x,y)
z = np.square(xx) - yy>0

print(z[1,1])
# for i in range(z.shape[0]):
#     for j in range(z.shape[1]):
#         if z[i,j]==True:
#             z[i,j]=10
#         else:
#             z[i,j]=2
# z = 1 if (np.square(xx) - yy).all() >0 else 0
# for i in range(10):
#     for j in range(z.shape[1]):
#         z[i,j]+=100
#         z[i+50,j]+=100
plt.contourf(xx,yy,z,cmap="cool")
plt.scatter(xx,yy,c = z)
plt.show()