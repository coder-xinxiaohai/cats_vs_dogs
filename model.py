import torch
import torch.nn as nn

# torch.nn是专门为神经网络设计的模块化接口。nn.Module是nn中最重要的类，可以把它看作一个网络的封装，
# 包含网络各层定义及forward方法，调用forward(input)方法，可以返回前向传播的结果。
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__() # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2), # input:[3, 224, 224]    output:[48, 55, 55]
            nn.ReLU(inplace=True), # 用输出的数据覆盖输入的数据，节省空间。
            nn.MaxPool2d(kernel_size=3, stride=2), # output:[48, 27, 27]

            nn.Conv2d(48, 128, kernel_size=5, padding=2), # output:[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # output:[128, 13, 13]

            nn.Conv2d(128, 192, kernel_size=3, padding=1), # output:[192, 13, 13]
            nn.ReLU(inplace=True),

            nn.Conv2d(192, 192, kernel_size=3, padding=1), # output:[192, 13, 13]
            nn.ReLU(inplace=True),

            nn.Conv2d(192, 128, kernel_size=3, padding=1), #output:[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2) # output:[128, 6, 6]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(), # 神经元随机失活的比例默认为0.5
            nn.Linear(128 * 6 * 6, 2048), # 全连接层
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2), # 猫狗分类为2分类问题。
        )

    # 只要在nn.Module的子类中定义了forward函数，backward函数就会被自动实现。
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1) # 展平处理(去掉batch维度)
        x = self.classifier(x)
        return x

# 在使用PyTorch封装好的网络层时，不需要对模型的参数初始化，因为这些PyTorch都会帮助我们完成。
# 但是如果是我们自己搭建模型，不使用pPyTorch中的封装好的网络层或者对PyTorch中封装好的模型初始化参数不满意，此时我们就需要对模型进行参数初始化。

# # 打印网络结构
# net = AlexNet()
# print(net)
#
# # net.parameters返回可学习的参数个数
# params = list(net.parameters())
# print(len(params))
#
# # net.named_parameters返回可学习的参数及名称
# for name, parameters in net.named_parameters():
#     print(name, ':', parameters.size())