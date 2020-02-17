
# 导入所有的包
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.optim as optim
from torchvision import transforms as tfs
from torchvision import models
from torch import nn
import matplotlib.pyplot as plt

# 实现数据预处理

train_transform = tfs.Compose([
    tfs.RandomResizedCrop(224),
    tfs.ToTensor(),
    tfs.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
])

test_transform = tfs.Compose([
    tfs.RandomResizedCrop(224),
    tfs.ToTensor(),
    tfs.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
])

batch_size = 64

train_set = ImageFolder('G:\\dataset\\distracted_driver_detection\\dataset\\train',train_transform)
train_data = DataLoader(train_set, batch_size, shuffle=True, num_workers=2)

valid_set = ImageFolder('G:\\dataset\\distracted_driver_detection\\dataset\\valid', test_transform)
valid_data = DataLoader(valid_set, batch_size, shuffle=False, num_workers=2)

train_valid_set = ImageFolder('G:\\dataset\\distracted_driver_detection\\dataset\\train_valid',train_transform)
train_valid_data = DataLoader(train_valid_set, batch_size, shuffle=True, num_workers=2)

# 测试
# =======不要修改这里的内容========
# try:
#     if iter(train_data).next()[0].shape[0] == batch_size and iter(train_data).next()[0].shape[0] == batch_size:
#         print('Success!')
#     else:
#         print('Not success, maybe the batch size is wrong!')
# except:
#     print('not success, image transform is wrong!')
    
# 构建模型，推荐使用 torchvision.models 中的预训练模型，将最后的全连接层修改成10分类
# 如果忘记了如何修改最后的全连接层，可以看看通过微调进行迁移学习的那个教程

def get_model():
    model = models.resnet34(pretrained=True)
    #预训练模型输出层Linear(in_features=512,#out_features=1000, bias=True)
    #需要修改fc输出类别为10
    model.fc = nn.Linear(512,10)
    return model

# 测试
# =======不要修改这里的内容========
# try:
#     model = get_model()
#     print(model.fc)
#     with torch.no_grad():#不进行反向传播
#         score = model(Variable(iter(train_data).next()[0]))
#     if score.shape[0] == batch_size and score.shape[1] == 10:#这里shape[1]为类别
#         print('successed!')
#     else:
#         print('failed!')
# except:
#     print('model is wrong!')

model = get_model()
# 根据自己的情况修改是否使用GPU
use_gpu = True

if use_gpu:
    model = model.cuda()
    # 构建loss函数和优化器


criterion = nn.CrossEntropyLoss() # 使用交叉熵作为loss函数
optimizer = optim.Adam(model.parameters(), lr=0.0001) # 可以使用前面讲过的多种优化方式

# 训练的 epochs 数目
max_epoch = 20

# def evaluate_accuracy(data_iter,net,device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
#     acc_sum,n = 0.0,0
#     with torch.no_grad():
#         for X,y in data_iter:
#             if isinstance(net,torch.Moudle):#判断类型
#                 net.eval()
def train(model, train_data, valid_data, max_epoch, criterion, optimizer):
    # 开始训练
    freq_print = int(len(train_data) / 3)

    metric_log = dict()
    metric_log['train_loss'] = list()
    metric_log['train_acc'] = list()
    if valid_data is not None:
        metric_log['valid_loss'] = list()
        metric_log['valid_acc'] = list()
    for e in range(max_epoch):
        model.train()
        running_loss = 0  #训练损失
        running_acc = 0  
        batch_count = 0
        for i, data in enumerate(train_data, 1):
            img, label = data
            if use_gpu:
                img = img.cuda()
                label = label.cuda()
            # 网络前向传播
            img_hat = model(img)
            # 计算误差
            loss = criterion(img_hat,label)
            # 反向传播，更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.cpu ().item()#一次batch的平均误差
            # 计算准确率,这里running_acc表示分类准确个数
            running_acc += (img_hat.argmax(dim=1)==label).sum().cpu().item()
            batch_count+=label.shape[0]
            if i % freq_print == 0:
                print('[{}]/[{}], train loss: {:.3f}, train acc: {:.3f}'.format(
                    i, len(train_data), running_loss / i, running_acc / (i*batch_size)))

        metric_log['train_loss'].append(running_loss / batch_count)
        metric_log['train_acc'].append(running_acc / batch_count)
        
        #训练完一个epoch,进行一次测试
        if valid_data is not None:
            model.eval()  #进 入评估测试
            running_loss = 0
            running_acc = 0
            batch_count = 0
            for data in valid_data:
                img, label = data
                if use_gpu:
                    img = img.cuda()
                    label = label.cuda()
                # 网络前向传播
                img_hat = model(img)
                # 计算误差
                loss =criterion(img_hat,label)
                running_loss += loss.cpu().item()
                batch_count+=label.shape[0]
                # 计算准确率
                running_acc += (img_hat.argmax(dim=1)==label).float().sum().cpu().item()
            metric_log['valid_loss'].append(running_loss / batch_count)
            metric_log['valid_acc'].append(running_acc / batch_count)
            print_str = 'epoch: {}, train loss: {:.3f}, train acc: {:.3f}, \
                valid loss: {:.3f}, valid accuracy: {:.3f}'.format(
                e + 1, 
                metric_log['train_loss'][-1], 
                metric_log['train_acc'][-1], 
                metric_log['valid_loss'][-1], 
                metric_log['valid_acc'][-1]
                )
            model.train()

        else:
            print_str = 'epoch: {}, train loss: {:.3f}, train acc: {:.3f}'.format(
                e + 1, 
                metric_log['train_loss'][-1],
                metric_log['train_acc'][-1])
        print(print_str)
        print()
    # =======不要修改这里的内容========
    # 可视化
    nrows = 1
    ncols = 2
    figsize= (10, 5)
    _, figs = plt.subplots(nrows, ncols, figsize=figsize)
    if valid_data is not None:
        figs[0].plot(metric_log['train_loss'], label='train loss')
        figs[0].plot(metric_log['valid_loss'], label='valid loss')
        figs[0].axes.set_xlabel('loss')
        figs[0].legend(loc='best')
        figs[1].plot(metric_log['train_acc'], label='train acc')
        figs[1].plot(metric_log['valid_acc'], label='valid acc')
        figs[1].axes.set_xlabel('acc')
        figs[1].legend(loc='best')
    else:
        figs[0].plot(metric_log['train_loss'], label='train loss')
        figs[0].axes.set_xlabel('loss')
        figs[0].legend(loc='best')
        figs[1].plot(metric_log['train_acc'], label='train acc')
        figs[1].axes.set_xlabel('acc')
        figs[1].legend(loc='best')
        
# 用作调参使用
if __name__ == '__main__':
    train(model, train_data, valid_data, max_epoch, criterion, optimizer)
    torch.save(model.state_dict(), './save_model.pth')#保存训练好模型的