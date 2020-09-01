# -*- coding:utf-8 -*-
from nnieqat import quant_dequant_weight, unquant_weight, merge_freeze_bn, register_quantization_hook
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class TestCifar10(unittest.TestCase):
    def test(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data',
                                                train=True,
                                                download=True,
                                                transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=4,
                                                  shuffle=True,
                                                  num_workers=2)
        testset = torchvision.datasets.CIFAR10(root='./data',
                                               train=False,
                                               download=True,
                                               transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=4,
                                                 shuffle=True,
                                                 num_workers=2)

        dataiter = iter(trainloader)
        images, labels = dataiter.next()
        net = Net()
        register_quantization_hook(net)
        net.cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


        print("Cifar10 training:")
        for epoch in range(5):
            net.train()
            if epoch > 2:
                net = merge_freeze_bn(net)
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = Variable(inputs.cuda()), Variable(
                    labels.cuda())
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                net.apply(unquant_weight)
                optimizer.step()

                running_loss += loss.item()
                if i % 2000 == 1999:
                    print(' epoch %3d, Iter %5d, loss: %.3f' %
                                (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
        print('Finished Training.')

        # net.apply(quant_dequant_weight)
        correct = total = 0
        for data in testloader:
            images, labels = data
            outputs = net(Variable(images.cuda()))
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels.cuda()).sum()
            total += labels.size(0)
        print(
            'Accuracy(10000 test images, modules\' weight unquantize): %d %%' %
            (100.0 * correct / total))


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(TestCifar10("test"))
    runner = unittest.TextTestRunner()
    runner.run(suite)
