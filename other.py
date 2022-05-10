"""学习过程中的一些代码记录，与猫狗分类项目本身并无太大关系。"""


"""验证nn.ReLU(inplace=True)中参数inplace=True的作用"""
# import torch
# import torch.nn as nn

# Relu = nn.ReLU(inplace=True)
# # 随机生成5个数 有正有负。
# input = torch.randn(5)
# # 打印 随机生成的数
# print(input)
#
# output = Relu(input)
# # 经过nn.ReLU()作用之后
# print(output)
# print(input)


# ————————————————————————————————
"""验证random_split()的作用"""
# import torch
# from torch.utils.data import random_split

# dataset = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
# print(len(dataset))
# torch.manual_seed(0)
# train_dataset,test_dataset = random_split(
#     dataset = dataset,
#     lengths =[6,7],
# )
# print(list(train_dataset))
# print(list(test_dataset))


# ————————————————————————————————
"""验证start=0和start=1的区别"""
# for i in range(3):
#     print(i)
# print(i)
#
# for step, data in enumerate([1, 2, 3, 4], start=1):
#     print('step:', step)
#     print('data:', data)
# print(step)


# ————————————————————————————
"""验证nn.Module中的forward函数"""
# import torch
# import torch.nn as nn
#
# class Model(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, input):
#         output = input + 1
#         print(output)
#
# model = Model()
# input = torch.tensor([1, 2, 3, 4, 5])
# model(input)


# -------------------------------------------------------
"""验证torch.max的作用"""
# import torch
# a = torch.tensor([[1,5,62,54], [2,6,2,6], [2,65,2,6]])
# print(a)
# predict = torch.max(a, dim = 1)
# print(predict)


# -------------------------------------------------------
"""验证torch.softmax的作用"""
# import torch
# a = torch.tensor([1., 100.])
# print(a)
# predict = torch.softmax(a, dim = 0)
# print(predict)


# -----------------------------------------------------
"""Pytorch是否使用GPU"""
# import torch
#
# print(torch.__version__)
# print(torch.cuda.is_available()) # 检查cuda是否可用
# print(torch.cudnn.is_available()) # 检查cudnn是否可用
#
# print(torch.cuda.current_device()) # 返回当前设备索引
# print(torch.cuda.device_count()) # 返回GPU的数量
# print(torch.cuda.get_device_name(0)) # 返回GPU的名字
