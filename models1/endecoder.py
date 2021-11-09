import torch.nn as nn
import math

class encoder(nn.Module):
    def __init__(self,length=60,lead=8,kernel=3,output_channel=16):
        super(encoder,self).__init__()

        self.length=length
        self.lead=lead
        self.kernel_size=kernel
        self.output_channel=output_channel
        self.conv1=nn.Conv1d(3,8,kernel_size=self.kernel_size,stride=1,padding=1)
        self.conv2=nn.Conv1d(8,self.output_channel,kernel_size=self.kernel_size,stride=2,padding=1)

        self.relu=nn.ReLU()
    def forward(self,x):
        x=self.conv1(x)
        x=self.relu(x)
        x=self.conv2(x)
        x=self.relu(x)
        embedding=[]
        for i in range(x.size(-1)):
            if  i+3<=x.size(-1):
                #squeezed_feature=nn.AdaptiveMaxPool1d(x,dim=1)
                #embedding.append(squeezed_feature)
                embedding.append(x[:,:,i:i+3])
            else:
                continue
        output=torch.stack(embedding,dim=0).permute(1,0,2,3)
        output=output.reshape((output.size(0),output.size(1),self.kernel_size*self.output_channel))
        return output #batch*length*channel*3

class decoder(nn.Module):
    def __init__(self,input,num_layer=1,hidden_size=64,fc_dim=32,classes=2):
        super(decoder, self).__init__()
        self.input_size=input
        self.num_layer=num_layer
        self.hidden_size=hidden_size
        self.fc_dim=fc_dim
        self.classes=classes

        self.LSTM=nn.LSTM(
            self.input_size,
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




if __name__ == '__main__':
    import torch

    x = torch.randn(10, 3, 60)


    encoder=encoder()
    out=encoder(x)
    decoder=decoder(out.size(-1))
    out=decoder(out)

    print(out)