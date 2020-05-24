#计算模型在toplogos数据集上的准确率
ListName = ['adidas0','chanel','gucci','hh','lacoste','mk','nike','prada','puma','supreme']
def computepre10(list1,num):
    TrueNum = ""
    FalseNum = ""
    for i in list1:
        TrueNum = sum(list1)
        FalseNum = num - sum(list1)
    return TrueNum,FalseNum
def getMap10():
    f = open("a1.txt")
    line = f.readline()
    line = line.strip('\n')
    line = line.strip('[')
    line = line.strip(']')
    line = line.split(",")
    num = -1
    list1 = []
    for i in line:
        num += 1
        if(num == 700):
            break
        list1.append(int(i))
    TrueNum,FalseNum = computepre32(list1,num)
    return  TrueNum,FalseNum
def computepre32(list1,num):
    TrueNum = ""
    FalseNum = ""
    for i in list1:
        TrueNum = sum(list1)
        FalseNum = num - sum(list1)
    return TrueNum,FalseNum
def getMap32():
    f = open("a.txt")
    line = f.readline()
    line = line.strip('\n')
    line = line.strip('[')
    line = line.strip(']')
    line = line.split(",")
    num = -1
    list1 = []
    for i in line:
        num += 1
        if(num == 2240):
            break
        list1.append(int(i))
    TrueNum,FalseNum = computepre32(list1,num)
    return  TrueNum,FalseNum
if __name__ == '__main__':
   TrueNum,FalseNum = getMap32()
   print("检测Flickrlogo32数据集的准确率",TrueNum/(2*(TrueNum+FalseNum)))
   TrueNum, FalseNum = getMap10()
   print("检测Toplogo10数据集的准确率", TrueNum / (2*(TrueNum + FalseNum)))




