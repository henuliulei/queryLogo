
"""
数据相关操作，包括数据预处理、dataset实现

"""
import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T
from PIL import Image
ListName = ['adidas0','chanel','gucci','hh','lacoste','mk','nike','prada','puma','supreme']
import cv2
# 目标图片集的处理
transform_img = T.Compose([
    T.Resize(256), # 缩放图片，保持长宽比不变，最短边为256
    T.CenterCrop(256), # 从图片中间切出224*224的图片
    T.ToTensor(), # 将图片转成Tensor，归一化至{0,1]
    T.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])  # 标准化至[-1,1]
])
# 查询logo图片集的处理
transform_logo = T.Compose([
    T.Resize(64), # 缩放图片，保持长宽比不变，最短边为256
    T.CenterCrop(64), # 从图片中间切出224*224的图片
    T.ToTensor(), # 将图片转成Tensor，归一化至{0,1]
    T.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])  # 标准化至[-1,1]
])
# mask图片集的处理
transform_mask = T.Compose([
    T.Resize(256),  # 缩放图片，保持长宽比不变，最短边为256
    T.CenterCrop(256),] # 从图片中间切出256*256的图片
)


def get_mask(root):
    # root:D:\baiduNetLoad\FlickrLogos-32_dataset_v2\FlickrLogos-v2\classes\jpg\adidas\79242964.jpg
    # mask_PNG:D:\baiduNetLoad\FlickrLogos-32_dataset_v2\FlickrLogos-v2\classes\masks\adidas\79242964.jpg.mask.0.png
    mask_path = root.replace('jpg','masks',1).replace('.jpg','.jpg.mask.merged.png')#把图片路径换成对应的mask路径
    mask_img = Image.open(mask_path)

    mask_t = transform_mask(mask_img)
    mask = np.array(mask_t).astype(np.float32)
    mask[mask>0] = 255
    return mask  # [256,256]

class FLogo(data.Dataset):
    def __init__(self,data_dir,transform1 = transform_img,transform2 = transform_logo,train=True,test=False):
        """
        获取所有图片地址，并根据训练、验证、测试划分数据
        :param data_dir: 图片数据集
        :param transforms:  图片预处理
        :param Train: 训练集？
        :param test: 测试集？
        """
        self.test = test
        self.train = train
        self.transform1 = transform1
        self.transform2 = transform2
        if self.test:
            imgPath_list_file = os.path.join(data_dir, 'testset.relpaths.txt')
            img_paths = [os.path.join(data_dir, id_).strip().replace('/', '\\') for id_ in open(imgPath_list_file)]
        elif train:
            imgPath_list_file = os.path.join(data_dir, 'all_paths_cc.txt')
            img_paths = [os.path.join(data_dir, id_).strip().replace('/', '\\') for id_ in open(imgPath_list_file)]
        else:
            imgPath_list_file = os.path.join(data_dir, 'valset.relpaths.txt')
            img_paths = [os.path.join(data_dir, id_).strip().replace('/', '\\') for id_ in open(imgPath_list_file)]#？？

        self.img_paths = img_paths
        # self.logo_paths = [img_path.replace('jpg','crop',1) for img_path in img_paths]


    def __getitem__(self, index):
        '''
        返回一张图片的数据
        如果是测试集，统一返回-1？
        :param item:
        :return:
        '''
        img_path = self.img_paths[index]
        logo_path = self.img_paths[index].replace('jpg','crop',1)

        if self.test:
            label = np.ones([256,256])*(-1)
        else:
            label = get_mask(img_path)

        data1 = Image.open(img_path)
        data2 = Image.open(logo_path)
        if self.transform1:
            data1 = self.transform1(data1)
        if self.transform2:
            data2 = self.transform2(data2)

        return data1,data2,label   # 显然是不完全满足要求的，因为要有两个输入，也就是这里要包含query logo的

    def __len__(self):
        '''
        :return: 数据集中所有图片的个数
        '''
        return len(self.img_paths)
#切分类名和图片名
def Spilt(line):
    className=""
    picName=""
    flag = 0
    for i in line:
        #print(i)
        if i!=" " and flag==0:
            className += i
        elif i==" " and flag==0:
            flag=1
        elif i!=" " and flag==1 and i!='\n':
            picName += i
    return className,picName
basepath = "C:/Users/19575/Desktop/study/bishe/code1/queryLogo"
def gasuss_noise(image, mean=0, var=0.001):
  image = np.array(image/255, dtype=float)
  noise = np.random.normal(mean, var ** 0.5, image.shape)
  out = image + noise
  if out.min() < 0:
    low_clip = -1.
  else:
    low_clip = 0.
  out = np.clip(out, low_clip, 1.0)
  out = np.uint8(out*255)
  #cv.imshow("gasuss", out)
  return out
def GetTestLogoset():#获取FlickrLogos-32_datasets数据集里的数据内容
    f = open(basepath+"/dataset/FlickrLogos-32_dataset_v2/FlickrLogos-v2/all.spaces.txt")
    basequery = basepath+"/dataset/FlickrLogos-32_dataset_v2/FlickrLogos-v2/classes/crop/"
    basetarget = basepath+"/dataset/FlickrLogos-32_dataset_v2/FlickrLogos-v2/classes/jpg/"
    basemasks = basepath+"/dataset/FlickrLogos-32_dataset_v2/FlickrLogos-v2/classes/masks/"

    queryList = []
    targetList = []
    masklist= []
    #line= f.readline()
    num = 0
    pathtargetList = [] #目标图片路径
    gtPath = []#获取每张图片groundtruth的路劲
    maskpath = "D:\\PIC\\FlickrLogos-v2\\classes\\masks"
    while True:
        num += 1
        line = f.readline()
        if num == 8241:
            break
        classname,picname = Spilt(line)
        if classname == "no-logo":
            continue
        pathquery = basequery + classname + "/" +picname
        pathtarget = basetarget + classname + "/" + picname
        pathmasks = basemasks + classname + "/" + picname + ".mask.0.png"
        gtPath.append(maskpath + "\\" +classname + "\\" +  picname + ".bboxes.txt")
        pathtargetList.append(pathtarget)
        query = Image.open(pathquery)
        target = Image.open(pathtarget)
        target = gasuss_noise(np.array(target))
        mask = Image.open(pathmasks)
        query = transform_logo(query)
        target = transform_img(Image.fromarray(target))
        mask = transform_mask(mask)
        queryList.append(query)
        targetList.append(target)
        masklist.append(mask)

    f.close()
    return queryList,targetList,masklist,pathtargetList,gtPath
def GetTestLogoset1():#获取Toplogos10数据集里的数据内容
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    f = open(basepath + "/dataset/qmul_toplogo10/ImageSets/all.txt")
    basequery = basepath + "/dataset/qmul_toplogo10/query/"
    basetarget = basepath + "/dataset/qmul_toplogo10/jpg/"
    originmasksData =basepath + "/dataset/qmul_toplogo10/masks/"
    queryList = []
    targetList = []
    # line= f.readline()
    num = -1
    pathtargetList = []  # 目标图片路径
    gtPath = []  # 获取每张图片groundtruth的路劲
    while True:
        num += 1
        line = f.readline()
        line = line.strip("\n")
        if num == 700:
            break
        pathquery = basequery + ListName[int(num/70)] + "/" + line + ".jpg"
        pathtarget = basetarget + ListName[int(num/70)] + "/" + line + ".jpg"
        gtPath.append(originmasksData + ListName[int(num/70)] + "/" + line + ".jpg.bboxes.txt")
        pathtargetList.append(pathtarget)
        query = Image.open(pathquery)
        target = Image.open(pathtarget)
        print(num)
        target = gasuss_noise(np.array(target))
        query = transform_logo(query)
        target = transform_img(Image.fromarray(target))
        queryList.append(query)
        targetList.append(target)
    f.close()
    return queryList, targetList, pathtargetList, gtPath
def getImg(pathquery,pathtarget):
    query = Image.open(pathquery)
    target = Image.open(pathtarget)
    target = gasuss_noise(np.array(target))
    query = transform_logo(query)
    target = transform_img(Image.fromarray(target))
    return query,target, pathtarget

'''
import matplotlib.pyplot as plt
if __name__ == '__main__':
    queryList, targetList,maskList = GetTestLogoset()
    print(queryList[0].shape)
    print(type(queryList[0]))
    plt.imshow(queryList[0])
    plt.show()
    plt.imshow(targetList[0])
    plt.show()
    plt.imshow(maskList[0])
    plt.show()
'''