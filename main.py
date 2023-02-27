import numpy as np
import torch 
from build_net import ZeroShotNet
import torchvision.models as models
from build_dataset import ZeroShotDataset
import torch.nn as nn
import matplotlib.pyplot as plt
import sklearn.manifold as fold
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F


def main():

    train_classes = ['airplane', 'automobile', 'bird', 'cat',
                     'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    test_classes = train_classes + ['bicycle', 'helicopter', 'submarine']
    
    #对训练集及测试集数据的不同处理组合
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_test = transforms.Compose([     
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    

    # CIFAR-10 数据集下载
    cifar_train_dataset = torchvision.datasets.CIFAR10(root='./data/',
                                                train=True, 
                                                transform=transform_train,
                                                download=False)

    cifar_test_dataset = torchvision.datasets.CIFAR10(root='./data/',
                                                train=False, 
                                                transform=transform_test,
                                                download=False)
    
    train_dataset = ZeroShotDataset(cifar_train_dataset, train_classes, '.')
    test_dataset = ZeroShotDataset(cifar_test_dataset, test_classes, '.')


    # 数据载入
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=32,
                                            shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=1,
                                            shuffle=False)
    
    # model = ZeroShotNet()

    model = models.resnet50(pretrained=True)

    # for param in model.parameters():
    #     param.requires_grad = False
    # 最后一层全连接层
    fc_inputs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(fc_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 100)
    )
    print(model)


    #定义损失函数，分类问题使用交叉信息熵，回归问题使用MSE
    criterion = nn.CrossEntropyLoss()
    #torch.optim来做算法优化,该函数甚至可以指定每一层的学习率，这里选用Adam来做优化器，还可以选其他的优化器
    optimizer = optim.Adam(model.parameters(),lr=0.0001)
    EPOCH = 20
    #设置GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    for epoch in range(EPOCH):
        for i,data in enumerate(train_loader):
            #取出数据及标签
            inputs,labels = data
            #数据及标签均送入GPU或CPU
            inputs,labels = inputs.to(device),labels.to(device)
            model.to(device)
            
            #前向传播
            outputs = model(inputs)

            outputs = torch.matmul(outputs, train_dataset.embedding_metric.to(device).T)
            # soft = nn.Softmax(dim=0)

            #计算损失函数
            loss = criterion((outputs),labels)
            #清空上一轮的梯度
            optimizer.zero_grad()
            
            #反向传播
            loss.backward()
            #参数更新
            optimizer.step()

            #print('it’s training...{}'.format(i))
            print('epoch{} loss:{:.4f}'.format(epoch+1,loss.item()))

    #保存模型参数
    torch.save(model,'cifar10_zero_shot.pt')
    print('cifar10_zero_shot.pt saved')




if __name__ == '__main__':
    main()