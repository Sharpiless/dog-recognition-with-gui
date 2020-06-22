# 基于 ResNet50 的狗狗品种识别

# 联系我们：
权重文件需要的请私戳作者~

联系我时请备注所需模型权重，我会拉你进交流群~

该群会定时分享各种源码和模型，之前分享过的请从群文件中下载~

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200613141749103.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)

# 1. 效果预览：
（似乎有奇怪的东西混进去了）

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200516083105525.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200516083121828.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200516083140690.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200516083202409.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200516083219703.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)

# 2. ResNet算法详解：
这个我写过一个更详细的：

[残差神经网络ResNet系列网络结构详解：从ResNet到DenseNet](https://blog.csdn.net/weixin_44936889/article/details/103774753)

这里简单复述一下：

## 2.1 论文地址：
[《Deep Residual Learning for Image Recognition》](https://arxiv.org/pdf/1512.03385v1.pdf)

## 2.2 核心思想：
将本来回归的目标函数H(x)转化为F(x)+x，即F(x) = H(x) - x，称之为残差。

## 2.3 网络结构：
### 2.3.1 残差单元：
ResNet的基本的残差单元如图所示：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191230230520573.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)

基本结构如图，假设每个单元输入的特征层为x，经过两个卷积层获得输出y，将x与y求和即得到了这个单元的输出；

在训练时，我们将该单元目标映射（即要趋近的最优解）假设为F(x) + x，而输出为y+x，那么训练的目标就变成了使y趋近于F(x)。即去掉映射前后相同的主体部分x，从而突出微小的变化（残差）。

用数学表达式表示为：
![在这里插入图片描述](https://img-blog.csdnimg.cn/201912302311531.png)

其中：
1. x是残差单元的输入；
2. y是残差单元的输出；
3. F(x)是目标映射；
4. {Wi}是残差单元中的卷积层；
5. Ws是一个1x1卷积核大小的卷积，作用是给x降维或升维，从而与输出y大小一致（因为需要求和）；

### 2.3.2 改进单元：

同时也可以进一步拓展残差结构：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191230230951707.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)

原论文中则以VGG为例：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191230231833962.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)

从VGG的19层，拓展到了34层。

可见使用了残差单元可以大大加深卷积神经网络的深度，而且不会影响性能和训练速度.

## 2.4 实现代码：
传送门：
[ResNet-tensorflow](https://github.com/Sharpiless/RESNET-tensorflow)

残差单元的实现：

```python
# block1
net = slim.repeat(res, 2, slim.conv2d, 64, [3, 3],
                  scope='conv1', padding='SAME')
res = net

# block2
net = slim.repeat(res, 2, slim.conv2d, 64, [3, 3],
                  scope='conv2', padding='SAME')

net = tf.add(net, res) # y=F(x)+x

```

## 2.5 实验结果：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191230232906419.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)
在ImageNet数据集上的测试表明，随着层数的加深，ResNet取得的效果越来越好，有效解决了模型退化的和梯度消失的问题。

# 3. 数据集简介：
数据地址：[https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)

数据集由 UDACITY 提供，包含超过8000张、共133种不同品种的狗狗图像：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200516084345743.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200516084437962.png)
下载之后解压，并创建类别名的ch_names.txt文件：

```bash
猴㹴
阿富汗猎犬
万能㹴
秋田犬
阿拉斯加雪橇犬
萨摩耶
美国猎狐犬
美国斯塔福郡㹴
美国水猎犬
安纳托利亚牧羊犬
澳洲牧牛犬
澳大利亚牧羊犬
澳大利亚㹴
巴辛吉犬
巴吉度猎犬
比格犬
长须牧羊犬
法兰西野狼犬
贝林登㹴
比利时玛连莱犬
比利时狗牧羊犬
比利时特伏丹犬
伯恩山犬
比熊犬
黑棕棕猎犬
黑俄罗斯㹴
猎犬
布鲁泰克浣熊猎犬
边境牧羊犬
边境㹴
苏俄牧羊犬
波士顿㹴
法兰德斯牧羊犬
拳师犬
帕金猎犬
伯瑞犬
布列塔尼猎犬
布鲁塞尔格里丰犬
斗牛犬
斗牛㹴
斗牛獒
凯恩㹴
迦南犬
意大利卡斯罗犬
意大利卡柯基犬
骑士查理王猎犬
乞沙比克猎犬
吉娃娃
中国冠毛犬
中国沙皮犬
松狮
克伦伯猎犬
可卡犬
牧羊犬
卷毛寻回猎犬
腊肠犬
大麦町斑点狗
丹迪丁蒙㹴
杜宾犬
波尔多獒犬
可卡犬
英文字母
英国雪达蹲猎犬
英国玩赏犬
恩特雷布赫山地犬
田野猎犬
芬兰猎犬
平滑毛寻回犬
法国斗牛犬
德平犬
德国牧羊犬
德国短毛指示犬
德国钢毛指示犬
巨型雪纳瑞犬
峡谷㹴
金毛寻回犬
戈登塞特犬
大丹犬
大比利牛斯犬
大瑞士山狗
灵缇犬
哈瓦那犬
伊比沙猎犬
冰岛牧羊犬
爱尔兰红白塞特犬
爱尔兰塞特犬
爱尔兰㹴
爱尔兰水猎犬
爱尔兰猎犬
意大利灵缇犬
日本狆
荷兰毛狮犬
凯利蓝㹴
可蒙犬
库瓦兹犬
拉布拉多犬
莱克兰㹴
伦伯格犬
拉萨阿普索犬
罗秦犬
马尔济斯犬
曼彻斯特㹴
獒
迷你雪纳瑞犬
那不勒斯犬
纽芬兰犬
诺福克犬
挪威布恩德犬
挪威埃尔克猎犬
挪威隆德犬
诺里奇㹴
新斯科舍猎犬
英国古代牧羊犬 
奥达猎犬
蝴蝶犬 
帕森罗素㹴
狮子狗
彭布罗克威尔士柯基
迷你贝吉格里芬凡丁犬 
法老王猎犬
布劳特猎犬
英国指示犬
博美犬
贵宾犬
葡萄牙水犬
圣伯纳犬 
澳洲丝毛㹴
短毛猎狐㹴
藏獒
威尔士史宾格犬
刚毛指示格里芬犬
墨西哥无毛犬
约克夏犬
```

# 4. 训练模型：
不想耗时间训练的可以联系我，然后下载我训练好的模型；

自己训练的话，运行 train.py：
```python
# train.py
import os
import torch
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, target)
            loss.backward()
            
            optimizer.step()
            
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            
            if batch_idx% 100 == 0:
                print('Epoch %d, Batch %d loss: %.6f' %
                  (epoch, batch_idx + 1, train_loss))
        ######################    
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        
        ## TODO: save the model if validation loss has decreased
        if valid_loss < valid_loss_min:
            torch.save(model,save_path)
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            valid_loss_min = valid_loss
    # return trained model
    return model


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    print("use_cuda",use_cuda)
    batch_size = 20
    num_workers = 0
    data_directory = 'dogImages/'
    train_directory = os.path.join(data_directory, 'train/')
    valid_directory = os.path.join(data_directory, 'valid/')
    test_directory = os.path.join(data_directory, 'test/')



    standard_normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
    data_transforms = {'train': transforms.Compose([transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        standard_normalization]),
                    'valid': transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        standard_normalization]),
                    'test': transforms.Compose([transforms.Resize(size=(224,224)),
                                        transforms.ToTensor(), 
                                        standard_normalization])
                    }

    train_data = datasets.ImageFolder(train_directory, transform=data_transforms['train'])
    valid_data = datasets.ImageFolder(valid_directory, transform=data_transforms['valid'])
    test_data = datasets.ImageFolder(test_directory, transform=data_transforms['test'])

    train_loader = torch.utils.data.DataLoader(train_data,
                                            batch_size=batch_size, 
                                            num_workers=num_workers,
                                            shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data,
                                            batch_size=batch_size, 
                                            num_workers=num_workers,
                                            shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data,
                                            batch_size=batch_size, 
                                            num_workers=num_workers,
                                            shuffle=False)
    loaders = {
        'train': train_loader,
        'valid': valid_loader,
        'test': test_loader
    }

    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs,133) 
    if use_cuda:
        model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.0001)  

    n_epochs = 10

    train(n_epochs, loaders, model, optimizer, criterion, use_cuda, 'model.pt')
```


训练完把模型文件放到src文件夹下；

然后下载字体文件，放到src文件夹下：

链接：[https://pan.baidu.com/s/1zt2DoqFUDaHphb-TX8UiJQ](https://pan.baidu.com/s/1zt2DoqFUDaHphb-TX8UiJQ) 
提取码：ko0t

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200516090243509.png)

# 5. 测试图片：
创建以下程序 net.py：

```python
from pylab import mpl
import cv2
import os
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import warnings
import numpy as np

from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

fontC = ImageFont.truetype('./src/platech.ttf', 32, 0)

warnings.filterwarnings("ignore")

mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


def drawText(image, addText, x1, y1):

    # mean = np.mean(image[:50, :50], axis=0)
    # color = 255 - np.mean(mean, axis=0).astype(np.int)
    # color = tuple(color)
    color = (20, 255, 20)
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    draw.text((x1, y1),
              addText.encode("utf-8").decode("utf-8"),
              color, font=fontC)
    imagex = np.array(img)

    return imagex


class Classifier(object):

    def __init__(self):

        self.model = torch.load('./src/model.pt')
        self.model.eval()
        # self.class_names = torch.load('names.class')
        self.loader = transforms.Compose([transforms.Resize((224, 224)),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.485, 0.456, 0.406),
                                                               (0.229, 0.224, 0.225))])
        with open('./src/ch_names.txt', 'r', encoding='utf-8') as f:
            self.class_names = f.read().splitlines()

    def test(self, pt):

        image = Image.open(pt)
        image_tensor = self.loader(image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        inputs = Variable(image_tensor)

        output = self.model(inputs)
        softmax = nn.Softmax(dim=1)
        preds = softmax(output)
        top_preds = torch.topk(preds, 3)
        pred_breeds = [self.class_names[i] for i in top_preds[1][0]]
        confidence = top_preds[0][0]

        result = self.DrawResultv2(pred_breeds, confidence, pt)

        return result

    def DrawResultv2(self, breeds, confidence, pt, size=512):
        title = "预测品种(置信度):\n"
        for breed, conf in zip(breeds, confidence):
            if conf > 0.005:
                title += "  - {} ({:.0f}%)\n".format(breed, conf*100)
        image = cv2.imdecode(np.fromfile(pt, dtype=np.uint8), 1)
        w = min(image.shape[:2])
        ratio = size / w
        image = cv2.resize(image, None, fx=ratio, fy=ratio)
        image = drawText(image, title, 0, 10)

        print(title)
        return image


if __name__ == '__main__':

    pt = './test_images/eee.jpg'
    net = Classifier()
    result = net.test(pt)
    cv2.imshow('a', result)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

```

将 pt 改成测试图片的路径即可：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200516090549383.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)

测试效果：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020051609064284.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)

# 联系我们：
权重文件需要的请私戳作者~

联系我时请备注所需模型权重，我会拉你进交流群~

该群会定时分享各种源码和模型，之前分享过的请从群文件中下载~

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200613141749103.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)

