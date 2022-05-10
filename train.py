import os,sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torch.utils.data import random_split
from torchvision import transforms
from PIL import Image
from model import AlexNet


# 数据集路径
train_path = './data/train_and_validate'

if torch.cuda.is_available(): # 检查cuda可用
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    # torch.device代表将torch.Tensor分配到的设备对象，有cpu和gpu两种
    # 这里的cuda就是gpu，至于为什么不直接采用gpu与cpu对应，是因为gpu的编程接口采用的是cuda.
# print(device) # cuda

# 数据集预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

# 自定义数据集
class MyDataset(Dataset):
    def __init__(self, data_path, train=True, transform=transform):
        self.data_path = data_path
        self.train_flag = train
        self.transform = transform
        self.path_list = os.listdir(data_path) #列出所有图片的命名

    def _read_convert_image(self, image_path):
        image = Image.open(image_path)
        image = self.transform(image)  # 把图片转化为tensor
        return image

    def _read_image_label(self, image_name):
        if self.train_flag:
            if image_name.split('.')[0] == 'dog':
                label = 1
            else:
                label = 0
        else:
            label = int(image_name.split('.')[0]) # 获取测试集图片的编号
        label = torch.tensor(label) # 把标签转化为tensor
        return label

    def __getitem__(self, index): #支持一个整型索引来获取单个数据
        image_name = self.path_list[index]
        image_path = os.path.join(self.data_path, image_name)
        image = self._read_convert_image(image_path)
        label = self._read_image_label(image_name)
        return image,label

    def __len__ (self):
        images_num = len(self.path_list)
        return images_num

train_data = MyDataset(train_path)


# image_tensor = train_data[1][0]
# print(image_tensor.shape) #使用tensor.shape来检查张量的形状
#
# TensorToPIL = transforms.ToPILImage() # 将张量转化为PIL图片
# image = TensorToPIL(image_tensor)
#
# print(image.size)
# plt.imshow(image)
# plt.show()

# 将训练集划分为训练集和验证集
train_ratio, validate_ratio =  0.7, 0.3
len_data = len(train_data)
train_num = int(0.7 * len_data)
val_num = int(0.3 * len_data)
train_dataset, validate_dataset = random_split(
    dataset=train_data,
    lengths=[train_num, val_num]
)

# 数据分批
BATCH_SIZE = 20
train_loader = Data.DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle = True, # 随机打乱数据集。
    pin_memory=False, # 锁存内存。一般来说，在GPU训练时，设置为True，在CPU上设置为False。
                      # 由于pin_memory与电脑硬件性能有关，pytorch开发者不能确保每个炼丹玩家都有高端设备，因此pin_memory默认为False。
    drop_last=False # 是否将最后一个不足batch的数据丢弃，默认为False。
)
validate_loader = Data.DataLoader(
    dataset=validate_dataset,
    batch_size=BATCH_SIZE,
    shuffle = False, # 验证和测试一般不需要打乱数据集
    pin_memory=False,
    drop_last=False
)


# 创建一个AlexNet网络对象net
net = AlexNet()
# 将net交给GPU运行
net.to(device)
loss_function = nn.CrossEntropyLoss() # 计算交叉熵损失
# 在反向传播计算完所有参数的梯度后，还需要使用优化方法更新网络的权重和参数。
# torch.optim中实现了深度学习中绝大多数的优化方法，例如Adam。
# 新建一个优化器，设置学习率，并指定要调整的参数。
# # PyTorch将深度学习中常用的优化方法全部封装在torch.optim中，其设计十分灵活，能够很方便地扩展成自定义的优化方法。
# # 所有的优化方法都是继承基类optim.Optimizer，并实现了自己的优化步骤。
optimizer = optim.Adam(net.parameters(), lr=0.0002)
save_path = './AlexNet.pth' # 保存模型
best_acc = 0.0 # 10次训练过程中的最佳精度

# 迭代10次
for epoch in range(10):
    net.train() # 训练阶段用net.train()
    running_loss = 0.0 # 训练阶段一个batch的平均损失
    t1 = time.perf_counter() # 返回当前计算机系统的时间
    # 循环训练集 从1开始
    for step, data in enumerate(train_loader, start = 1):
        images, labels = data # data是一个列表，[数据，标签]
        optimizer.zero_grad() # 优化器的梯度清0，每次循环都需要清零，否则梯度会无限叠加，相当于增加批次大小。
        outputs = net(images.to(device)) # 将输入的数据分配到指定的GPU中
        loss = loss_function(outputs, labels.to(device))
        loss.backward() # loss进行反向传播
        optimizer.step() # step进行参数更新
        running_loss += loss.item() # item()返回loss的值，每次计算完loss后加入到running_loss中，可以算出叠加之后的总loss
    print(time.perf_counter()-t1) # 记录训练一个epoch所需要的时间

    net.eval() # 非训练阶段用net.eval()
    acc = 0.0
    with torch.no_grad():
    # 在使用PyTorch时，并不是所有的操作都需要进行计算图的生成(计算过程的构建，以便梯度反向传播等操作)
    # 而对于tensor的计算操作，默认是要进行计算图的构建的，在这种情况下，可以使用with.torch.no_grad():
    # 强制之后的内容不进行计算图的构建
        for data_validate in validate_loader:
            validate_images, validate_labels = data_validate
            outputs = net(validate_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1] # dim=1表示行 dim=0表示列 网络预测为对应类别的概率 而行代表样本 列代表类别
            # 输出每行最大值所对应的索引，即计算模型中每个类别的最大值并返回其索引值，即该类别的标签值
            acc += (predict_y == validate_labels.to(device)).sum().item() # .sum就是将所有值相加，得到的仍然是tensor，使用.item()之后得到的就是值。
        acc_validate = acc/val_num
        if acc_validate > best_acc:
            best_acc = acc_validate
            torch.save(net.state_dict(), save_path) # 只保存网络中的参数

            # PyTorch保存模型与加载
            # 模型的保存
            # torch.save(net,PATH) # 保存模型的整个网络，包括网络的整个结构和参数
            # torch.save(net.state_dict, PATH) # 只保持网络中的参数
            # 模型的加载
            # 分别对应上边的加载方法
            # model_dict=torch.load(PATH)
            # model_dict=net.load.dict(torch.load(PATH)

        print('[epoch %d] train_loss: %.3f validate_accuracy: %.3f' %(epoch + 1, running_loss / step, acc / val_num))
print('Finish Training')

