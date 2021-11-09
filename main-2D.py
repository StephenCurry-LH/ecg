'''
训练模型，此处使用的网络是resnet34

'''
import torch, time, os, shutil
from tqdm import tqdm
import numpy as np
import pywt, os, copy
import torch
import pandas as pd
from torch.utils.data import Dataset
from scipy import signal
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from dateset_hf import DatasetECG
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import *

def validation(model,validation_loader,device,optimizer,epoch):
    model.eval()
    losses = []
    scores = []
    N_count = 0
    interval = 100
    with torch.no_grad():
        for batch_idx,(x,y) in enumerate(validation_loader):
            x=x.to(device).permute(0,2,1).float()  #
            y=y.to(device).float()
            N_count+=y.size(0)
            output=model(x)
            criterion = nn.BCEWithLogitsLoss()
            loss = criterion(output.squeeze(), y)
            out = F.sigmoid(output)
            prediction = [0 if item < 0.5 else 1 for item in out.squeeze()]
            correct = [1 if prediction[idx] == y[idx] else 0 for idx in range(len(prediction))]
            losses.append(loss.item())
            scores.extend(correct)
            if batch_idx % interval == 1:
                print('validation epoch:{},loss:{:.5f},accuracu:{}'.format(epoch, sum(losses) / N_count,
                                                                         sum(scores) / N_count))
                losses = []
                scores = []
                N_count = 0

def train(model, train_loader, device, optimizer, epoch):
    model.train()
    losses = []
    scores = []
    N_count = 0
    interval=100
    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.to(device).permute(0, 2, 1).float()  #batch*5000*8，设置第一个卷积核为8通道，使得与8导联匹配
        y = y.to(device).float()  # shape= （batch）

        N_count += y.size(0)
        optimizer.zero_grad()
        output = model(x)  # shape=（batch * k）
        # print(output.squeeze().shape,output.squeeze().shape)
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(output.squeeze(), y)
        out = F.sigmoid(output)
        # print(out, y, loss)
        prediction=[0 if item <0.5 else 1 for item in out.squeeze()]
        correct=[1 if prediction[idx]==y[idx] else 0 for idx in range(len(prediction))]

        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        scores.extend(correct)
        if batch_idx % interval==1:
            print('training epoch:{},loss:{:.5f},accuracu:{}'.format(epoch,sum(losses)/N_count,sum(scores)/N_count))
            losses = []
            scores = []
            N_count = 0









if __name__ == '__main__':
    all_data = []
    all_label = []
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    all_yes_data = os.listdir('./data1/yes')
    all_no_data = os.listdir('./data1/no')
    all_yes_label = [1] * len(all_yes_data)
    all_no_label = [0] * len(all_no_data)
    all_data.extend(all_yes_data)
    all_data.extend(all_no_data)
    all_label.extend(all_yes_label)
    all_label.extend(all_no_label)
    x_train, x_test, y_train, y_test = train_test_split(all_data, all_label, test_size=0.3, random_state=42)  # 随机分解数据集
    params = {'batch_size': 128, 'shuffle': True, 'num_workers': 16, 'pin_memory': True}
    train_set = DatasetECG('./data1/all', x_train, y_train)
    train_loader = DataLoader(train_set, **params)
    validation_set = DatasetECG('./data1/all', x_test, y_test)
    test_loader = DataLoader(validation_set, **params)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    k = 1  # classes num
    # 以上为数据汇总及处理


    epochs=100
    learning_rate=0.001
    model=resnet152(num_classes=1).to(device) #resnet152()


    optimiezer=torch.optim.Adam(model.parameters(),learning_rate)
    for epoch in tqdm(range(epochs)):
        train(model,train_loader,device,optimiezer,epoch)
        validation(model,train_loader,device,optimiezer,epoch)


