import torch
import torchvision
import torch.nn as nn
from torch.utils import data
from torchvision import transforms
from torch.autograd import Variable
from tensorboardX import SummaryWriter

'''定义参数'''
batch_size = 64
lr = 0.001
num_classes = 10
writer = SummaryWriter(log_dir='scalar')
# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式
# 并除以255使得所有像素的数值均在0到1之间
# 图片转换为tensor
trans = transforms.ToTensor()

'''获取数据集'''
# 训练集
mnist_train = torchvision.datasets.MNIST(
    root="./data", train=True, transform=trans, download=True)
# 测试集
mnist_test = torchvision.datasets.MNIST(
    root="./data", train=False, transform=trans, download=True)

'''装载数据'''
train_loader = data.DataLoader(mnist_train, batch_size, shuffle=True)
test_loader = data.DataLoader(mnist_test, batch_size, shuffle=False)

class LeNet(nn.Module):
    def __init__(self, num_class=10):
        super().__init__() # 继承nn.model类
        '''第一层卷积，卷积核大小为5*5，步距为1，输入通道为1，
        输出通道为6(feature map),padding = 2（32*32->28*28）'''
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        '''第一层池化层，卷积核为2*2，步距为2'''
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        '''第二层卷积，卷积核大小为5*5，步距为1，输入通道为6，输出通道为16'''
        self.conv2=nn.Conv2d(6, 16, kernel_size=5, stride=1)
        '''第二层池化层，卷积核为2*2，步距为2'''
        self.pool2=nn.MaxPool2d(kernel_size=2, stride=2)
        '''第一层全连接层，维度由16*5*5=>120'''
        self.linear1 = nn.Linear(16 * 5 * 5, 120)
        '''第二层全连接层，维度由120=>84'''
        self.linear2 = nn.Linear(120, 84)
        '''第三层全连接层，维度由84=>10'''
        self.linear3 = nn.Linear(84, num_classes)
    def forward(self,x):
        out = torch.tanh(self.conv1(x))
        out = self.pool1(out)
        out = torch.tanh(self.conv2(out))
        out = self.pool2(out)
        out = out.reshape(-1, 16*5*5)
        out = torch.sigmoid(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        out = self.linear3(out)
        return out

def do_train():
    model = LeNet(num_classes)
    model = model.cuda()
    '''设置损失函数'''
    criterion = nn.CrossEntropyLoss()
    '''设置优化器'''
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    '''开始训练'''
    total_step = len(train_loader)
    total_loss = 0
    for epoch in range(20):
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda()
            labels = labels.cuda()
            images = Variable(images)
            labels = Variable(labels)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss
            if (i + 1) % 300 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, 20, i + 1, total_step, loss.item()))
        writer.add_scalar("Train_Loss: ", total_loss/len(train_loader), epoch)
        writer.flush()
    # torch.save(model, './model/lenet5.pth')

def do_test():
    model = torch.load('./model/lenet5.pth')
    model.eval()
    model = model.cuda()

    '''设置损失函数'''
    criterion = nn.CrossEntropyLoss()
    '''设置优化器'''
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    correct = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.cuda()
            labels = labels.cuda()
            images = Variable(images)
            labels = Variable(labels)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # 获取预测结果的下标（即预测的数字）
            _, preds = torch.max(outputs, dim=1)
            # 累计预测正确个数
            correct += torch.sum(preds == labels)

    print('accuracy rate: ', correct/len(test_loader.dataset))
do_train()
do_test()
writer.close()