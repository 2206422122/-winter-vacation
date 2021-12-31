import torch
from torch import nn

#定义网络模型

class MyLeNet5(nn.Module):
    #初始化
    def __init__(self):
        ##继承父类
        super(MyLeNet5, self).__init__()

        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.sigmoid = nn.Sigmoid()
        self.s2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.s4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)

        self.flatten = nn.Flatten()#展开方便全连接
        self.f6 = nn.Linear(120, 84)#全连接层
        self.output = nn.Linear(84, 10)

    ##前向传播函数
    def forward(self, x):
        x = self.sigmoid(self.c1(x))
        x = self.s2(x)
        x = self.sigmoid(self.c3(x))
        x = self.s4(x)
        x = self.c5(x)
        x = self.flatten(x)
        x = self.f6(x)
        x = self.output(x)

        return x

if __name__ == "__main__":
    ##随机初始化权重张量
    x = torch.rand([1,1,28,28])
    ##实例化网络
    model = MyLeNet5()
    y = model(x)