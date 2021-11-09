import os
from scipy import signal
import numpy as np
from tqdm import tqdm

def get_label_num(label_name_file):  #统计每一个类别的数量
    with open(label_name_file,'r') as f:
        label_name=[]
        for line in f.readlines():
            label_name.append(line.split()[0])
    num=[int(0)]*len(label_name)
    label_file='./data1/hf_round1_label.txt'
    with open(label_file,'r') as f:
        for line in f.readlines():
            for label in line.split():
                if label in label_name:
                    idx=label_name.index(label)
                    num[idx]+=1

    return dict(zip(label_name,num))

#{'窦性心律': 16918, 'T波改变': 3421, '窦性心动过缓': 3372, 'ST段改变': 2967, '正常ECG': 4171, '左心室高电压': 4326, 'ST-T改变': 2111, '窦性心动过速': 2010, '临界ECG': 1911, 'QRS低电压': 1543, '房性早搏': 1470, '电轴左偏': 1137, '电轴右偏': 1055, '心房颤动': 1217, '异常ECG': 1061, '室性早搏': 971, '完全性右束支传导阻滞': 1109, '窦性心律不齐': 924, '左心室肥大': 432, '右束支传导阻滞': 392, '一度房室传导阻滞': 282, '快心室率': 229, '不完全性右束支传导阻滞': 199, '非特异性T波异常': 125, 'QT间期延长': 101, '左前分支传导阻滞': 106, '非特异性ST段异常': 78, '差异性传导': 75, '非特异性ST段与T波异常': 61, '起搏心律': 74, '短PR间期': 55, '下壁异常Q波': 53, '快室率心房颤动': 42, '逆钟向转位': 34, '室内差异性传导': 36, '早期复极化': 37, '二联律': 27, '室上性早搏': 33, '顺钟向转位': 30, '复极化异常': 29, '未下传的房性早搏': 25, '肺心病型': 27, '慢心室率': 30, '短串房性心动过速': 21, '非特异性室内传导延迟': 16, '右心房扩大': 24, '左束支传导阻滞': 18, '前间壁R波递增不良': 19, '右心室肥大': 18, '房室传导延缓': 10, '双分支传导阻滞': 20, '非特异性室内传导阻滞': 17, '肺型P波': 17, '完全性左束支传导阻滞': 20, '融合波': 18}
#结果
#Statistics_data=get_label_num('./data/hf_round1_arrythmia1.txt')
#以"窦性心律"为标准，做一个二分类测试频域效果

#处理8个导联的数据，txt转为npy格式
def read_ecg(file):
    with open(file,'r') as f:
        array=[]
        for line in f.readlines()[1:]: #从第二行开始计数
            line=[int(x) for x in line.split()]
            array.extend(line)
        array=np.array(array).reshape(5000,-1) #5000*8
    return array



#read_ecg('./data/train/2.txt')




 #将txt里的数据转为npy储存，并按照疾病分类
train_data_path='./data1/train'  #全部txt
label_path='./data1/hf_round1_label.txt'
dir_path='multi_class-3D'
if not os.path.exists(os.path.join('data',dir_path)):
    os.mkdir(os.path.join('data',dir_path))
#'./data/multi_class {标签}_{患者编号}.txt，保存格式
#Sinus rhythm 窦性心律
with open (label_path,'r') as f:
    for line in tqdm(f.readlines()):
        patient_name=line.split()[0]
        patient_num=os.path.splitext(patient_name)[0] #患者编号
        #type1='yes' if '窦性心律' in line.split() else 'no'
        if '窦性心律' in line.split():
            if 'QRS低电压' in line.split():
                disease='1'
            else:
                disease='2'
        else:
            disease='0'


        patient_name=os.path.join(train_data_path,patient_name)
        array=read_ecg(patient_name)

        npy_name=os.path.join('data',dir_path,disease+'_'+patient_num+'.npy')
        #print(npy_name)
        np.save(npy_name,array)


#find -name "*.npy" |xargs -i cp {} ./all






