from os.path import join
from tqdm import tqdm
import pandas as pd

import torchvision
import torch
import torch.nn as nn

#  设置device，使得网络和数据能够在gpu上运行
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#  定义卷积+激活函数类，定义方法要继承nn.Module
class Conv2dTanh(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding=(0, 0), bias=True):
        super(Conv2dTanh, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        """
        Args:
            x: [N,C,H,W]
        """
        o1 = self.conv(x)
        o2 = torch.tanh(o1)
        return o2


#  定义特征提取类
class Features(nn.Module):
    def __init__(self, padding):
        super(Features, self).__init__()

        self.padding = padding
        self.conv2dtanh1 = Conv2dTanh(in_channels=1, out_channels=6, kernel_size=5, padding=padding)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2dtanh2 = Conv2dTanh(in_channels=6, out_channels=16, kernel_size=5)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2)
        self.conv2dtanh3 = Conv2dTanh(in_channels=16, out_channels=120, kernel_size=5)

    def forward(self, x):
        """
        Args:
            x: [N,1,H,W]
            x: [Batchnum, 1, Height, Weight]
            x: [200, 1, 28, 28]
            o1: [200, 6, 28, 28]
            o2: [200, 6, 14, 14]
            o3: [200, 16, 10, 10]
            o4: [200, 16, 5, 5]
            o5: [200, 120, 1, 1]
        """
        o1 = self.conv2dtanh1(x)
        o2 = self.avgpool1(o1)
        o3 = self.conv2dtanh2(o2)
        o4 = self.avgpool2(o3)
        o5 = self.conv2dtanh3(o4)
        return o5


#  定义线性分类层,o3输出后没有定义激活函数，需要后定义softmax
class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()

        self.num_classes = num_classes

        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.fc2 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):
        """
        Args:
            x: [N,120]
        """
        o1 = self.fc1(x)
        o2 = torch.tanh(o1)
        o3 = self.fc2(o2)
        return o3


#  定义网络模型，调用上面方法类
class LeNet5(nn.Module):
    def __init__(self, num_classes=10, padding=0):
        super(LeNet5, self).__init__()

        self.num_classes = num_classes
        self.padding = padding

        self.features = Features(padding=padding)
        self.flatten = nn.Flatten()
        self.classifier = Classifier(num_classes=num_classes)

    def forward(self, x):
        """
        Args:
            x: [N,1,H,W]
            flatten: [Batchnum, channels, 1, 1]-> [Batchnum, channels]
        """
        o1 = self.features(x)
        o2 = self.flatten(o1)
        o3 = self.classifier(o2)
        #  softmax激活函数， dim=-1意味着dim=input.dim()
        o4 = torch.log_softmax(o3, dim=-1)
        return o4


#  定义数据集的读取方式dataloader，下载方式，方法已经封装好了
class Datasets:
    def __init__(self, dataset_path, batch_size):
        self.train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(dataset_path, train=True, download=True,
                                       transform=torchvision.transforms.ToTensor()),
            batch_size=batch_size, shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(dataset_path, train=False, download=True,
                                       transform=torchvision.transforms.ToTensor()),
            batch_size=batch_size * 2, shuffle=True)


#  训练方法类，包含训练，测试
class Trainer:
    def __init__(self, datasets, model, optimizer, loss_fn, results_path='results'):
        self.datasets = datasets
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.results_path = results_path
        self.train_df = None

    def train_epoch(self, msg_format):
        self.model.train()

        losses = []
        bar = tqdm(self.datasets.train_loader)
        for data, target in bar:
            #  将数据放入device显卡上
            data, target = data.to(device), target.to(device)
            #  把梯度.grad置零，如果不将梯度清零的话，梯度会与上一个batch的数据相关
            self.optimizer.zero_grad()

            output = self.model(data)
            loss = self.loss_fn(output, target)

            #  梯度自动计算，累加到.grad属性上(Batch)
            loss.backward()
            #  通过梯度下降法更新网络参数权重
            self.optimizer.step()
            #  设置loss值，val值等信息的显示格式
            bar.set_description(msg_format.format(loss.item()))
            #  记录loss信息
            losses.append(loss.item())
        return losses

    def test(self):
        #  防止网络更改参数，否则的话，有输入数据，即使不训练，它也会改变权值。这是model中含有batch normalization层所带来的的性质。
        self.model.eval()

        count = len(self.datasets.test_loader.dataset)
        test_loss = 0
        correct = 0
        #  被with torch.no_grad()包住的代码，不用跟踪反向梯度计算，即梯度函数状态不会更新
        with torch.no_grad():
            for data, target in self.datasets.test_loader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                test_loss += self.loss_fn(output, target).item() * len(data)
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum().item()

        return test_loss / count, correct / count

    def train(self, num_epoch):
        val_loss, accuracy = self.test()
        all_losses = [[None, val_loss, accuracy]]

        for epoch in range(num_epoch):
            # train
            train_losses = self.train_epoch(
                f'train {epoch}/{num_epoch} -- loss: {{:3.2f}}, val_loss: {val_loss:3.2f}, accuracy: {accuracy:.1%}')

            # test
            val_loss, accuracy = self.test()
            all_losses.extend([
                [train_loss, None, None]
                for train_loss in train_losses
            ])
            all_losses.append([None, val_loss, accuracy])

        self.save_model()
        #  定义pandas数据分析
        self.train_df = pd.DataFrame(data=all_losses, columns=["train_loss", "val_loss", "accuracy"])
        self.train_df.to_csv(join(self.results_path, "train.csv"), index=False)

    #  定义保存模型，这里在windows下可能无法生产results文件夹，自行创建
    def save_model(self):
        torch.save(self.model.state_dict(), join(self.results_path, 'model.pth'))

    def plot(self):
        import matplotlib.pyplot as plt
        self.train_df[["train_loss", "val_loss"]].ffill().plot(grid=True, logy=True)
        self.train_df[["accuracy"]].dropna().plot(grid=True)
        plt.show()


def train():
    torch.manual_seed(0)
    #  将网络模型放在device显卡上
    model = LeNet5(num_classes=10, padding=2).to(device)
    #  NLL:负对数似然
    loss_fn = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    trainer = Trainer(Datasets("datasets", 100), model=model, optimizer=optimizer,
                      loss_fn=loss_fn, results_path="results")

    trainer.train(num_epoch=3)
    trainer.plot()


if __name__ == "__main__":
    train()
