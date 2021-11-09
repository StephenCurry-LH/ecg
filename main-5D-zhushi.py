
'''
三分类有病或者没病，global class=1
多分了，global classes = 3,4,5、、、、、、、、
'''
import torch, time, os, shutil
from tqdm import tqdm
import pywt, os, copy
import torch
from scipy import signal
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from dateset_hf_multi import DatasetECG
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import *
from models.endecoder import *
from torch.utils.tensorboard import SummaryWriter

writer=SummaryWriter()


def validation(model,validation_loader,device,optimizer,epoch):
    model.eval()#model.train()和model.eval()的区别主要在于Batch Normalization和Dropout两层
    # 如果模型中有BN和Dropout，需要在训练时添加model.train（）。保证BN层能够用到每一批数据的均值和方差。对于Dropout，model.train()是随机取一部分网络连接来训练更新参数
    # model.eval()
    # 是保证BN层能够用全部训练数据的均值和方差，即测试过程中要保证BN层的均值和方差不变。对于Dropout，model.eval()
    # 是利用到了所有网络连接，即不进行随机舍弃神经元。
    # 训练完train样本后，生成的模型model要用来测试样本。在model(test)之前，需要加上model.eval()，否则的话
    # 有输入数据，即使不训练，它也会改变权值。这是model中含有BN层和Dropout所带来的的性质。
    losses = []
    scores = []
    N_count = 0
    with torch.no_grad():  # 不会求导
        for batch_idx, (x, y) in enumerate(validation_loader):  # enumerate：同时给出数据和数据下标
            x = x.to(device).permute(0, 2, 1).float()  # permut转置矩阵
            y = y.to(device)

            N_count += y.size(0)
            output = model(x)

            if global_classes <= 2:
                criterion = nn.BCEWithLogitsLoss()  # 二分类损失函数采用BCEWithLogitsLoss
                loss = criterion(output.squeeze(), y)  # squeeze对数据的维度进行压缩
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

            writer.add_scalar('Loss/val', loss.item(), global_step)
            writer.add_scalar('Learning rate/val', optimizer.param_groups[0]['lr'], global_step)
            losses.append(loss.item())
            scores.extend(correct)  # 对了几个

        print('validation epoch:{},loss:{:.5f},accuracu:{}'.format(epoch, sum(losses) / N_count,  # loss的和/总数
                                                                   sum(scores) / N_count))  # 预测对的个数/总数
        writer.add_scalar('val/Accuracy', sum(scores) / N_count, global_step)
        print('{}Epoch {} Finished{}'.format("*" * 20, epoch, "*" * 20))


def train(model, train_loader, device, optimizer, epoch):
    global global_step
    global global_classes
    model.train()
    losses = []
    scores = []
    N_count = 0
    interval = 40

    for batch_idx, (x, y) in enumerate(train_loader):

        x = x.to(device).permute(0, 2, 1).float()  # batch*5000*8，设置第一个卷积核为8通道，使得与8导联匹配
        y = y.to(device)
        N_count += y.size(0)

        optimizer.zero_grad()
        output = model(x)  # shape=（batch * k）

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

        writer.add_scalar('Loss/Train', loss.item(), global_step)
        writer.add_scalar('Learning rate/train', optimizer.param_groups[0]['lr'], global_step)
        global_step += 1

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        scores.extend(correct)

        if batch_idx % interval == (interval - 1):
            print(
                'training epoch:{},loss:{:.5f},accuracu:{}'.format(epoch, sum(losses) / N_count, sum(scores) / N_count))
            writer.add_scalar('Accuracy', sum(scores) / N_count, global_step)
            scheduler.step(sum(scores) / N_count)  # optimizer.step()通常用在每个mini-batch之中，而scheduler.step()通常用在epoch里面
            # 只有用了optimizer.step()，模型才会更新，而scheduler.step()是对lr进行调整
            losses = []
            scores = []
            N_count = 0


if __name__ == '__main__':

    global_step = 0  # 写入tensorboard的全局次数
    global_classes = 3  # 1

    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

    dir_path = './data/multi_class-3D'
    all_data = os.listdir(dir_path)
    all_label = [npy[:npy.index('_')] for npy in all_data]

    x_train, x_test, y_train, y_test = train_test_split(all_data, all_label, test_size=0.3, random_state=42)  # 随机分解数据集
    print("the number of all_data is ")
    print(len(all_data))
    print("the number of train is ")
    print(len(x_train))
    print("the number of test is ")
    print(len(x_test))
    # print(Counter(x_train))
    params = {'batch_size': 128, 'shuffle': True, 'num_workers': 16, 'pin_memory': True}

    train_set = DatasetECG(dir_path, x_train, y_train)
    train_loader = DataLoader(train_set, **params)

    validation_set = DatasetECG(dir_path, x_test, y_test)
    test_loader = DataLoader(validation_set, **params)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 以上为数据汇总及处理

    ckpt_path = './ckpt/resnet-3D'
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)
    epochs = 10
    learning_rate = 0.001

    model = resnet50(num_classes=global_classes).to(device)  # resnet152() 1-D卷积分类网络
    # model=seq2seq(encoder=encoder().to(device),decoder=decoder(num_classes=global_classes).to(device))  #编解码器网络

    optimiezer = torch.optim.Adam(model.parameters(), learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiezer, 'max', patience=2)

    for epoch in range(epochs):
        train(model, train_loader, device, optimiezer, epoch)
        validation(model, train_loader, device, optimiezer, epoch)
        torch.save(model.state_dict(), os.path.join(ckpt_path, 'epoch_{}.pth'.format(epoch)))
#dataset
# design model
# loss and optimizer
# train(
#     output = model()
#     loss = critersion(outputs,labels)

#     optimizer.zero_grad()梯度归零
#     loss.backward()反向传播求梯度
#     optimizer.step()参数更新
# )