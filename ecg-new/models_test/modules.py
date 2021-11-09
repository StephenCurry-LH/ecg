from math import cos,pi,sqrt
import torch
import torch.nn as nn

def get_freq_indices(method):
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [i for i in range(64)]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x

class MultiSpectralDCTLayer(nn.Module):
    def __init__(self,channel,length_feature,reduction=16,freq_sel_method = [0]):
        super(MultiSpectralDCTLayer,self).__init__()

        self.reduction=reduction
        self.length_feature=length_feature

        mapper_x=freq_sel_method


        self.num_split=len(mapper_x)

        #print('ori mapper',mapper_x)
        mapper_x = [temp_x * (length_feature // 64) for temp_x in mapper_x]
        #print('new mapper',mapper_x)

        self.dec_layer=DCTLayer(self.length_feature,mapper_x,channel)

        self.fc=nn.Sequential(
            nn.Linear(channel,channel//self.reduction,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//self.reduction,channel,bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        b,c,l=x.shape
        x_pooled=x
        y=self.dec_layer(x_pooled)
        y=self.fc(y).view(b,c,1)
        return x*y.expand_as(x)

class DCTLayer(nn.Module):
    def __init__(self,length,mapper_x,channel):
        super(DCTLayer,self).__init__()
        assert channel % len(mapper_x)==0
        self.num_freq=len(mapper_x)
        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(length,  mapper_x, channel))

    def build_filter(self,pos,freq,POS):
        result = cos(pi*freq*(pos+0.5)/POS)/sqrt(POS)
        return result if freq==0 else result*sqrt(2)

    def get_dct_filter(self,length_feature,mapper_x,channel):
        dct_filter=torch.zeros(channel,length_feature)
        c_part=channel//len(mapper_x)
        for i,u_x in enumerate(mapper_x):
            for t_x in range(length_feature):
                dct_filter[i*c_part:(i+1)*c_part,t_x]=self.build_filter(t_x,u_x,length_feature)
        return dct_filter

    def forward(self,x):

        #print(x.shape,self.weight.shape)
        x=x*self.weight
        result=torch.sum(x,dim=-1)
        return result

#
# if __name__=='__main__':
#     input_tensor=torch.rand(4,64,49)
#     #tensor = torch.rand(4, 64, 3, 3)
#     model = MultiSpectralDCTLayer(64,49, reduction=16, freq_sel_method='top4')
#     out=model(input_tensor)
#     print(input_tensor)
#     print(out)