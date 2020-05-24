#在Flickrlogos数据集上进行检测
from config import opt
from models.QueryLogo import Net
from data.dataset import GetTestLogoset
from data.dataset import getImg
import torch
import numpy as np
import cv2
global result
import matplotlib.pyplot as plt
def getLoc2(line):
    x = ""
    y = ""
    width = ""
    height = ""
    flag = 0
    for i in line:
        if i==" ":
            flag += 1
            continue
        if flag == 0 and i!=" ":
            x += i
        if flag == 1 and i != " ":
            y += i
        if flag == 2 and i != " ":
            width += i
        if flag == 3 and i != " ":
            height += i
    w = int(x)+int(width)
    z = int(y)+int(height)

    return int(x),int(y),w,z
def compute_iou(rec1, rec2): #rect1 = [0, 0, 21, 21](top, left, bottom, right)，计算交并比
    areas1 = (rec1[3] - rec1[1]) * (rec1[2] - rec1[0])
    areas2 = (rec2[3] - rec2[1]) * (rec2[2] - rec2[0])
    left = max(rec1[1],rec2[1])
    right = min(rec1[3],rec2[3])
    top = max(rec1[0], rec2[0])
    bottom = min(rec1[2], rec2[2])
    w = max(0, right-left)
    h = max(0, bottom-top)
    return w*h/(areas2+areas1-w*h)
def getBbox(mask):
    #mask = cv2.imread(maskPath)

    index = np.argwhere(np.array(mask) > 0)
    #print(index)
    if len(index) == 0:
        return 100,100,101,101 #对于预测错误的mask直接返回一个像素点
    else:
        xMin = index[:,1].min()
        xMax = index[:,1].max()
        yMin = index[:,0].min()
        yMax = index[:,0].max()
        width = xMax - xMin
        height = yMax - yMin
        #处理可能出现的误判点或者错误点，根据mask的面积和gt的面积比值判断，mask面积看成为1的像素点的个数和，两者比值取0.3作为标准决定更新宽高
        while  len(index)/(width * height) < 0.3:
            xMin += width*0.1
            yMin += height*0.1
            xMax -= width*0.1
            yMax -= height*0.1
            width -= width*0.2
            height -= height*0.2
        return xMin, yMin, xMax, yMax
def drawBBox(mask,imgpath):
    #mask = cv2.imread(maskpath)
    target = cv2.imread(imgpath)
    #cv2.imshow('detection',target)
    #cv2.waitKey(0)
    #print(np.array(mask).shape)
    maskWidth = len(mask[0])
    maskHeight = len(mask)
    targetWidth = len(target[0])
    targetHeight = len(target)
    xMin, yMIn, xMax, yMax = getBbox(mask)
    if (xMax - xMin) == 1:
        return -1, -1, -1, -1
    else:
        xMin = xMin * (targetWidth/maskWidth) #把预测的坐标值映射到原图里面去
        yMin = yMIn * (targetHeight/maskHeight)
        xMax = xMax * (targetWidth/maskWidth)
        yMax = yMax * (targetHeight/maskHeight)
        return xMin, yMin, xMax, yMax
def Totest():
    #device = torch.device('cuda') if opt.use_gpu else torch.device('cpu')
    # 模型

    net = Net()
    net.eval()
    if opt.load_model_path:
        net.load_state_dict(torch.load(opt.load_model_path),False)
    queryList, targetList, maskList,pathtargetList,gtpath = GetTestLogoset()
    predictResult = []
    for i in range(0,len(queryList)):
        print(i)
        queryList[i] = torch.tensor(np.expand_dims(queryList[i],axis = 0))
        targetList[i] = torch.tensor(np.expand_dims(targetList[i], axis=0))
        output = net(queryList[i],targetList[i])
        output = output.squeeze()
        predict = torch.sigmoid(output)
        #np.set_printoptions(threshold=100000)
        predict[predict > 0.5] = 1
        predict[predict <= 0.5] = 0
        xMin, yMin, xMax, yMax=drawBBox(predict.detach().numpy(),pathtargetList[i])
        #获取图片的groundtruth
        ft = open(gtpath[i])
        ft.readline()
        gtline = ft.readline()
        ft.close()
        xmin, ymin, xmax, ymax = getLoc2(gtline)
        rect1 = [xMin,yMin,xMax,yMax]
        rect2 = [xmin,ymin,xmax,ymax]
        a = compute_iou(rect1, rect2)
        if a > 0.5:
            predictResult.append(1)
        else:
            predictResult.append(0)
        #print(predict.detach().numpy())
        #predictMaskList.append(predict)
    return predictResult
def getMask(pathquery,pathtarget):
    net = Net()
    net.eval()
    if opt.load_model_path:
        net.load_state_dict(torch.load(opt.load_model_path), False)
        query, target, pathtarget = getImg(pathquery, pathtarget)
        query = torch.tensor(np.expand_dims(query, axis=0))
        target = torch.tensor(np.expand_dims(target, axis=0))
        output = net(query, target)
        output = output.squeeze()
        predict = torch.sigmoid(output)
        # np.set_printoptions(threshold=100000)
        predict[predict > 0.5] = 1
        predict[predict <= 0.5] = 0
    return predict
def getPredict(pathquery,pathtarget):
    net = Net()
    net.eval()
    if opt.load_model_path:
        net.load_state_dict(torch.load(opt.load_model_path), False)
        query, target, pathtarget = getImg(pathquery,pathtarget)
        query = torch.tensor(np.expand_dims(query, axis=0))
        target= torch.tensor(np.expand_dims(target, axis=0))
        output = net(query, target)

        output = output.squeeze()
        predict = torch.sigmoid(output)
        # np.set_printoptions(threshold=100000)
        predict[predict > 0.5] = 1
        predict[predict <= 0.5] = 0
        print(predict)
        plt.imshow(predict.detach().numpy())
        plt.show()
        xMin, yMin, xMax, yMax = drawBBox(predict.detach().numpy(), pathtarget)
    return xMin, yMin, xMax, yMax
if __name__ == '__main__':
    #output = getMask("D:\\graduateStudent\\ABBProject\\egnnModel\\data\\liulei\\test\\image_1paper_cup.jpg","D:\\graduateStudent\\ABBProject\\egnnModel\\data\\liulei\\train\\imgs\\image_1.jpg")
    #cv2.imshow('img',output.detach().numpy())
    #cv2.waitKey(0)
     result = Totest()
     f1 = open('a.txt','w')
     f1.write(str(result))
     f1.close()




