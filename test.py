from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import AlexNet
import torch

# 数据集预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

image = Image.open('./data/cat.jpg')
plt.imshow(image)

image = transform(image)
# print(image.shape) # torch.Size([3, 224, 224])

image = torch.unsqueeze(image, dim=0) # 增加了一个batch维度
# print(image.shape) # torch.Size([1, 3, 224, 224])

model = AlexNet()
model_weight_path = 'AlexNet.pth'
model.load_state_dict(torch.load(model_weight_path))

model.eval()
with torch.no_grad():
    # print('model(image).shape:',model(image).shape) # model(image).shape: torch.Size([1, 2])
    output = torch.squeeze(model(image)) # 压缩batch维度
    # print('output.shape:',output.shape) # output.shape: torch.Size([2])
    predict = torch.softmax(output, dim=0) # 行归一化(即归一化的维度为行) softmax后就是一个概率分布
    # print('predict:',predict) # predict: tensor([0.9233, 0.0767])
    predict_class = torch.argmax(predict).numpy() # 返回概率最大值所对应的索引值
if predict_class == 0:
    print("This is a cat!")
else:
    print("This is a dog!")
plt.show()
