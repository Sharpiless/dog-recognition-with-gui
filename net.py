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

    pt = './test_images/aaa.jpg'
    net = Classifier()
    result = net.test(pt)
    cv2.imshow('a', result)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
