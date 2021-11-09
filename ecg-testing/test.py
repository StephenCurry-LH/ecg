
import torch
a=torch.tensor(
              [
                  [1, 5, 5, 2],
                  [9, -6, 2, 8],
                  [-3, 7, -9, 10]

              ])
b=torch.argmax(a,dim=1)

x=torch.tensor([
    [0.1,0.2,0.7],
    [0.8,0.1,0.1]
])
print(torch.argmax(x,dim=1))