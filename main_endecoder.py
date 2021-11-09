
'''
二分类有病或者没病，global class=1
多分了，global classes = 3,4,5、、、、、、、、
'''
import torch, time, os, shutil
from tqdm import tqdm
import pywt, os, copy
import torch
from scipy import signal
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from dateset_hf import DatasetECG
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import *
from models.endecoder import *
from torch.utils.tensorboard import SummaryWriter

writer=SummaryWriter()



def validation(model,validation_loader,device,optimizer,epoch):
    model.eval()
    losses = []
    scores = []
    N_count = 0
    with torch.no_grad():
        for batch_idx,(x,y) in enumerate(validation_loader):
            x=x.to(device).permute(0,2,1).float()  #
            y=y.to(device)
            N_count+=y.size(0)
            output=model(x)
            if global_classes <= 2:
                criterion = nn.BCEWithLogitsLoss()
                loss = criterion(output.squeeze(), y)
                out = F.sigmoid(output)
                # print(out, y, loss)
                prediction = [0 if item < 0.5 else 1 for item in out.squeeze()]
                correct = [1 if prediction[idx] == y[idx] else 0 for idx in range(len(prediction))]
            else:
                criterion = nn.CrossEntropyLoss()
                loss = criterion(output, y)
                out = F.softmax(output, dim=1)
                prediction = torch.argmax(out, dim=1, keepdim=False)
                correct = [1 if prediction[idx] == y[idx] else 0 for idx in range(len(prediction))]

            losses.append(loss.item())
            scores.extend(correct)

        print('validation epoch:{},loss:{:.5f},accuracu:{}'.format(epoch, sum(losses) / N_count,
                                                                         sum(scores) / N_count))
        #writer.add_scalar('Validation Accuracy', sum(scores) / N_count, global_step)
        print('{}Epoch {} Finished{}'.format("*"*20,epoch,"*"*20))


def train(model, train_loader, device, optimizer, epoch):
    global global_step
    global global_classes
    model.train()
    losses = []
    scores = []
    N_count = 0
    interval=40
    for batch_idx, (x, y) in enumerate(train_loader):

        x = x.to(device).permute(0, 2, 1).float()  #batch*5000*8，设置第一个卷积核为8通道，使得与8导联匹配
        y = y.to(device) # shape= （batch）
        N_count += y.size(0)
        optimizer.zero_grad()
        output = model(x)  # shape=（batch * k）

        if global_classes<=2:
            criterion = nn.BCEWithLogitsLoss()
            loss = criterion(output.squeeze(), y)
            out = F.sigmoid(output)
            # print(out, y, loss)
            prediction=[0 if item <0.5 else 1 for item in out.squeeze()]
            correct=[1 if prediction[idx]==y[idx] else 0 for idx in range(len(prediction))]
        else:
            criterion = nn.CrossEntropyLoss()
            loss=criterion(output,y)
            out=F.softmax(output,dim=1)
            prediction=torch.argmax(out,dim=1,keepdim=False)
            correct=[1 if prediction[idx]==y[idx] else 0 for idx in range(len(prediction))]

        writer.add_scalar('Loss/Train', loss.item(), global_step)
        writer.add_scalar('Learning rate', optimizer.param_groups[0]['lr'], global_step)
        global_step += 1
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        scores.extend(correct)
        if batch_idx % interval == (interval - 1):
            print(
                'training epoch:{},loss:{:.5f},accuracu:{}'.format(epoch, sum(losses) / N_count, sum(scores) / N_count))
            writer.add_scalar('Training Accuracy', sum(scores) / N_count, global_step)
            scheduler.step(sum(scores) / N_count)
            losses = []
            scores = []
            N_count = 0

if __name__ == '__main__':

    global_step = 0  #写入tensorboard的全局次数
    global_classes=3  #1

    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

    dir_path = './data/multi_class'
    all_data = os.listdir(dir_path)
    all_label = [npy[:npy.index('_')] for npy in all_data]
    x_train, x_test, y_train, y_test = train_test_split(all_data, all_label, test_size=0.3, random_state=42)  # 随机分解数据集
    params = {'batch_size': 128, 'shuffle': True, 'num_workers': 16, 'pin_memory': True}

    train_set = DatasetECG(dir_path, x_train, y_train)
    train_loader = DataLoader(train_set, **params)
    validation_set = DatasetECG(dir_path, x_test, y_test)
    test_loader = DataLoader(validation_set, **params)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 以上为数据汇总及处理

    ckpt_path='./ckpt/endecoder'
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)
    epochs=100
    learning_rate=0.001

    #model=resnet18(num_classes=global_classes).to(device) #resnet152() 1-D卷积分类网络
    model=seq2seq(encoder=encoder().to(device),decoder=decoder(num_classes=global_classes).to(device))  #编解码器网络

    optimiezer=torch.optim.Adam(model.parameters(),learning_rate)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimiezer,'max',patience=2)

    for epoch in range(epochs):
        train(model,train_loader,device,optimiezer,epoch)
        validation(model,train_loader,device,optimiezer,epoch)
        torch.save(model.state_dict(),os.path.join(ckpt_path,'epoch_{}.pth'.format(epoch)))

