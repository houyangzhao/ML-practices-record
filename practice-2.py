#Review
#第一次完整的编写整个网络框架，使用的是数据集：CIFAR-10
#训练70个epoch后early stop
#在训练集上的准确率达到82.710%
#训练时长大概是40min
#感觉模型的构建不是很有品味，也许更深但更窄的网络会更好。没有使用maxpool,因为感觉全卷积的网络表现会更好
#loss function: CrossEntropyLoss
#optimizer: Adam,learning rate=0.0002
#early stop: 10个epoch没有提升就停止训练
#使用tensorboard记录训练过程
#使用nvitop监控GPU使用情况
#尚未解决的问题：gpu的占用率一直不是很高，怀疑和loader的num_workers有关，
#设置num_workers=0时gpu占用率30%左右，设置为4时gpu占用率呈波浪式，一会儿gpu为0，然后达到90%，然后又降到0

import os
import shutil
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter


log_dir = './sdla_cifar'
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)

writer = SummaryWriter(log_dir)
device = "cuda" if torch.cuda.is_available() else "cpu" 
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
batch_size = 128
train_data=torchvision.datasets.CIFAR10(root="./data",train=True,download=True,transform=transform_train)
test_data=torchvision.datasets.CIFAR10(root="./data",train=False,download=True,transform=transform_test)

trainloader=torch.utils.data.DataLoader(train_data,batch_size=batch_size,shuffle=True,num_workers=0)
testloader=torch.utils.data.DataLoader(test_data,batch_size=batch_size,shuffle=False,num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

class model(nn.Module):
    def __init__(self):
        super(model,self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(3,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(128*32*32, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    def forward(self,x):
        x=self.model(x)
        return x

model=model()
model=model.to(device)


criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.0002)
train_loss_list=[]
test_loss_list=[]
def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx % 50 == 0:
            print('Step: %d | Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (batch_idx, train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    train_loss_list.append(train_loss/batch_size)

def test(epoch):
    global best_acc,early_stop_counter
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    acc = 100.*correct/total
    if acc > best_acc:
        early_stop_counter = 0
        print('Saving..')
        torch.save(model.state_dict(), './model.pth')
        best_acc = acc
    else:
        early_stop_counter += 1
    test_loss_list.append(test_loss/batch_size)

if __name__ == '__main__':
    best_acc = 0
    early_stop_counter = 0
    for epoch in range(0, 4):
        train(epoch)
        test(epoch)
        writer.add_scalars('Loss Comparison', {
            'Train Loss': train_loss_list[-1],
            'Test Loss': test_loss_list[-1]
        }, epoch)
        if early_stop_counter >= 10:
            print("Early stopping triggered.")
            break
    writer.close()