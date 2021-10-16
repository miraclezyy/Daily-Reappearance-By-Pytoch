import torch
import torch.nn as nn
import torchvision as tv
from torch.autograd import Variable
import tqdm
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False

dir = './data'

noiseSize = 100  # 噪声维度
n_generator_feature = 64  # 生成器feature map数
n_discriminator_feature = 64  # 判别器feature map数
batch_size = 256
d_every = 1  # 每一个batch训练一次discriminator
g_every = 5  # 每五个batch训练一次generator


class NetGenerator(nn.Module):
    def __init__(self):
        super(NetGenerator, self).__init__()
        self.main = nn.Sequential(  # 神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行
            nn.ConvTranspose2d(noiseSize, n_generator_feature * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(n_generator_feature * 8),
            nn.ReLU(True),  # (n_generator_feature * 8) × 4 × 4        (1-1)*1+1*(4-1)+0+1 = 4
            nn.ConvTranspose2d(n_generator_feature * 8, n_generator_feature * 4, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(n_generator_feature * 4),
            nn.ReLU(True),  # (n_generator_feature * 4) × 8 × 8     (4-1)*2-2*1+1*(4-1)+0+1 = 8
            nn.ConvTranspose2d(n_generator_feature * 4, n_generator_feature * 2, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(n_generator_feature * 2),
            nn.ReLU(True),  # (n_generator_feature * 2) × 16 × 16
            nn.ConvTranspose2d(n_generator_feature * 2, n_generator_feature, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(n_generator_feature),
            nn.ReLU(True),  # (n_generator_feature) × 32 × 32
            nn.ConvTranspose2d(n_generator_feature, 3, kernel_size=5, stride=3, padding=1, bias=False),
            nn.Tanh()  # 3 * 96 * 96
        )

    def forward(self, input):
        return self.main(input)


class NetDiscriminator(nn.Module):
    def __init__(self):
        super(NetDiscriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, n_discriminator_feature, kernel_size=5, stride=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),  # n_discriminator_feature * 32 * 32
            nn.Conv2d(n_discriminator_feature, n_discriminator_feature * 2, kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(n_discriminator_feature * 2),
            nn.LeakyReLU(0.2, inplace=True),  # (n_discriminator_feature*2) * 16 * 16
            nn.Conv2d(n_discriminator_feature * 2, n_discriminator_feature * 4, kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(n_discriminator_feature * 4),
            nn.LeakyReLU(0.2, inplace=True),  # (n_discriminator_feature*4) * 8 * 8
            nn.Conv2d(n_discriminator_feature * 4, n_discriminator_feature * 8, kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(n_discriminator_feature * 8),
            nn.LeakyReLU(0.2, inplace=True),  # (n_discriminator_feature*8) * 4 * 4
            nn.Conv2d(n_discriminator_feature * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()  # 输出一个概率
        )

    def forward(self, input):
        return self.main(input).view(-1)  # 把所有数据一维输出


def train():
    for i, (image, _) in tqdm.tqdm(
            enumerate(dataloader)):  # type((image,_)) = <class 'list'>, len((image,_)) = 2 * 256 * 3 * 96 * 96
        real_image = Variable(image)
        real_image = real_image.cuda()

        if (i + 1) % d_every == 0:
            optimizer_d.zero_grad()
            output = Discriminator(real_image)  # 尽可能把真图片判为True
            print(output.size())
            error_d_real = criterion(output, true_labels)
            error_d_real.backward()

            noises.data.copy_(torch.randn(batch_size, noiseSize, 1, 1))
            fake_img = Generator(noises).detach()  # 根据噪声生成假图,detach()截断两个网络之间反向传播的连接
            fake_output = Discriminator(fake_img)  # 尽可能把假图片判为False
            error_d_fake = criterion(fake_output, fake_labels)
            error_d_fake.backward()
            optimizer_d.step()

        if (i + 1) % g_every == 0:
            optimizer_g.zero_grad()
            noises.data.copy_(torch.randn(batch_size, noiseSize, 1, 1))
            fake_img = Generator(noises)  # 这里没有detach
            fake_output = Discriminator(fake_img)  # 尽可能让Discriminator把假图片判为True
            error_g = criterion(fake_output, true_labels)
            error_g.backward()
            optimizer_g.step()


'''
def show(num):
    fix_fake_imags = Generator(fix_noises)
    fix_fake_imags = fix_fake_imags.data.cpu()[:64] * 0.5 + 0.5

    # x = torch.rand(64, 3, 96, 96)
    fig = plt.figure(1)

    i = 1
    for image in fix_fake_imags:
        ax = fig.add_subplot(8, 8, eval('%d' % i))
        # plt.xticks([]), plt.yticks([])  # 去除坐标轴
        plt.axis('off')
        plt.imshow(image.permute(1, 2, 0))
        i += 1
    plt.subplots_adjust(left=None,  # the left side of the subplots of the figure
                        right=None,  # the right side of the subplots of the figure
                        bottom=None,  # the bottom of the subplots of the figure
                        top=None,  # the top of the subplots of the figure
                        wspace=0.05,  # the amount of width reserved for blank space between subplots
                        hspace=0.05)  # the amount of height reserved for white space between subplots)
    plt.suptitle('第%d迭代结果' % num, y=0.91, fontsize=15)
    plt.show()
'''


def show(num):
    fix_fake_imags = Generator(fix_noises)
    fix_fake_imags = fix_fake_imags.data.cpu()[:1] * 0.5 + 0.5

    # x = torch.rand(64, 3, 96, 96)
    fig = plt.figure(1)

    i = 1
    for image in fix_fake_imags:
        plt.imshow(image.permute(1, 2, 0))
    plt.subplots_adjust(left=None,  # the left side of the subplots of the figure
                        right=None,  # the right side of the subplots of the figure
                        bottom=None,  # the bottom of the subplots of the figure
                        top=None,  # the top of the subplots of the figure
                        wspace=0.05,  # the amount of width reserved for blank space between subplots
                        hspace=0.05)  # the amount of height reserved for white space between subplots)
    plt.suptitle('第%d迭代结果' % num, y=0.91, fontsize=15)
    plt.show()


if __name__ == '__main__':
    transform = tv.transforms.Compose([
        tv.transforms.Resize(96),  # 图片尺寸, transforms.Scale transform is deprecated
        tv.transforms.CenterCrop(96),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 变成[-1,1]的数
    ])

    dataset = tv.datasets.ImageFolder(dir, transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                             drop_last=True)  # module 'torch.utils.data' has no attribute 'DataLoder'

    print('数据加载完毕！')

    Generator = NetGenerator()
    Discriminator = NetDiscriminator()

    optimizer_g = torch.optim.Adam(Generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(Discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion = torch.nn.BCELoss()

    true_labels = Variable(torch.ones(batch_size))  # batch_size
    fake_labels = Variable(torch.zeros(batch_size))
    fix_noises = Variable(torch.randn(batch_size, noiseSize, 1, 1))
    noises = Variable(torch.randn(batch_size, noiseSize, 1, 1))  # 均值为0，方差为1的正态分布

    if torch.cuda.is_available() == True:
        print('Cuda is available!')
        Generator.cuda()
        Discriminator.cuda()
        criterion.cuda()
        true_labels, fake_labels = true_labels.cuda(), fake_labels.cuda()
        fix_noises, noises = fix_noises.cuda(), noises.cuda()

    plot_epoch = [1, 5, 10, 20, 50, 100, 199]

    for i in range(200):  # 最大迭代次数
        train()
        print('迭代次数：{}'.format(i))
        if i in plot_epoch:
            show(i)
