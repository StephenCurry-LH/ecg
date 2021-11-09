import torch
from torch import nn
import os

class testModule(nn.Module):
    def __init__(self):
        super(testModule, self).__init__()
        self.g = nn.Conv2d(in_channels=1, out_channels=1,
                           kernel_size=1, stride=1, padding=0)
        self.operation_function = self._realOperation

    def forward(self, x):
        output = self.operation_function(x)
        return output

    def _realOperation(self, x):
        x = self.g(x)
        return x


class testModule2(nn.Module):
    def __init__(self):
        super(testModule2, self).__init__()
        self.g = nn.Conv2d(in_channels=1, out_channels=1,
                           kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.g(x)
        return x


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"
    input = torch.rand(4, 1, 1, 1).cuda()
    net = testModule()
    net2 = testModule2()
    gpu_num = torch.cuda.device_count()
    print('GPU NUM: {:2d}'.format(gpu_num))
    if gpu_num > 1:
        net = torch.nn.DataParallel(net, list(range(gpu_num))).cuda()
        net2 = torch.nn.DataParallel(net2, list(range(gpu_num))).cuda()
    out2 = net2(input)
    print(out2.size())
    out = net(input)
    print(out.size())
