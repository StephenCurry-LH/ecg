import torch.nn as nn
import math
from models_test.resnet import *
class encoder(nn.Module):
    def __init__(self,resnet_type=resnet18()):
        super(encoder,self).__init__()
        used_resnet=resnet_type
        modules = list(used_resnet.children())[:-1]
        self.resnet=nn.Sequential(*modules)
        self.relu=nn.ReLU()

    def forward(self,x):
        x=self.resnet(x) #batch*512
        # for i in range(x.size(-1)):
        #     if  i+3<=x.size(-1):
        #         #squeezed_feature=nn.AdaptiveMaxPool1d(x,dim=1)
        #         #embedding.append(squeezed_feature)
        #         embedding.append(x[:,:,i:i+3])  #方法二：序列化暂缓
            # else:
            #     continue
        #output=torch.stack(embedding,dim=0).permute(1,0,2,3)
        return x #batch*512*1

class decoder(nn.Module):
    def __init__(self,input=1,num_layer=1,hidden_size=16,fc_dim=8,num_classes=1):
        super(decoder, self).__init__()
        self.input_size=input
        self.num_layer=num_layer
        self.hidden_size=hidden_size
        self.fc_dim=fc_dim
        self.classes=num_classes

        self.LSTM=nn.LSTM(
            self.input_size, #输入特征尺寸
            self.hidden_size,
            self.num_layer,
            batch_first=True
        )
        self.fc=nn.Linear(self.hidden_size,self.fc_dim)
        self.fc_out=nn.Linear(self.fc_dim,self.classes)

    def forward(self,x):
        RNN_out,(h_n,h_c)=self.LSTM(x)
        RNN_laststep=RNN_out[:,-1,:]
        out=self.fc(RNN_laststep)
        out=self.fc_out(out)
        return out

class seq2seq(nn.Module):
    def __init__(self,encoder,decoder):
        super(seq2seq,self).__init__()
        self.encoder=encoder
        self.decoder=decoder

    def forward(self,x):
        return self.decoder(self.encoder(x))


if __name__ == '__main__':
    import torch
    x = torch.randn(10, 8, 128)
    encoder=encoder()
    decoder=decoder()
    seq2seq=seq2seq(encoder,decoder)
    out=seq2seq(x)
    print(out)