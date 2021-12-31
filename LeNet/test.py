import torch
from LeNet import MyLeNet5
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage

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

model.load_state_dict(torch.load('C:/Users/22064/Desktop/卷积网络搭建/LeNet/best+model.pth'))

#获取结果
classes = [
    '0',
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9',
]

#将张量转化为图片
show = ToPILImage()
for i in range(20):
    x, y = test_dataset[i][0], test_dataset[i][1]
    show(x).show()

    x = Variable(torch.unsqueeze(x, dim =0).float(), requires_grad=False).to(device)
    with torch.no_grad():
        pred = model(x)
        prediceted, actual = classes[torch.argmax((pred[0]))], classes[y]
        print(f'predicted:f"{prediceted}", actual:{actual}')