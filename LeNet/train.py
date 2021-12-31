import torch
from torch import nn
from LeNet import MyLeNet5
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import os

#图片数组形式转化为张量形式
data_transform = transforms.Compose([
    transforms.ToTensor()
])

#加载训练数据集
train_dataset = datasets.MNIST(root="./data", train=True, transform=data_transform, download=True)
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

#加载测试数据集
test_dataset = datasets.MNIST(root="./data", train=False, transform=data_transform, download=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)

#判断是否GPU训练
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#调用LeNet网络模型，转到GPU
model =  MyLeNet5().to(device)

#定义交叉熵损失函数
loss_fn= nn.CrossEntropyLoss()

#定义优化器：带动量的随机梯度下降
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

#学习率的调整
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

#定义数据集
def train(dataloader, modol, loss_fn, optimizer):
    loss, current, n = 0.0, 0.0, 0
    for batch, (x, y) in enumerate(dataloader):
        #前向
        x, y = x.to(device), y.to(device)
        output = modol(x)
        cur_loss = loss_fn(output, y)
        _, pred = torch.max(output, axis=1)
        #计算这一批次的精确度
        cur_acc = torch.sum(y == pred)/output.shape[0]

        #反向传播
        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()

        #这一轮累计精确度
        loss += cur_loss.item()
        current += cur_acc.item()
        n = n + 1
    print("训练误差" + str(loss/n))
    print("训练精确度" + str(current/n))

def val(dataloader, model, loass_fn):
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
        #前向
            x, y = x.to(device), y.to(device)
            output = model(x)
            cur_loss = loss_fn(output, y)
            _, pred = torch.max(output, axis=1)

            #计算这一批次的精确度
            cur_acc = torch.sum(y == pred)/output.shape[0]
            # 这一轮累计精确度
            loss += cur_loss.item()
            current += cur_acc.item()
            n = n + 1
        print("验证误差" + str(loss / n))
        print("验证精确度" + str(current / n))

        return current/n
#训练
epoch = 10
min_acc = 0

for t in range(epoch):
    print(f'epoch{t+1}\n----------------------')
    train(train_dataloader, model, loss_fn, optimizer)
    a = val(test_dataloader, model, loss_fn)

    #保存最优模型参数权重
    if a > min_acc:
        folder = 'save_model'
        if not os.path.exists(folder):
            os.mkdir('save_model')
        min_acc = a
        print("save best model")
        torch.save(model.state_dict(), 'best+model.pth')
    print('Done')

