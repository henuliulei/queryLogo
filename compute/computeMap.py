#计算模型在flickrlogo数据集上的map值
import matplotlib as plt
import cv2
import numpy as np
logoList = ["google","apple","adidas","HP","stellaartois","paulaner","guiness","singha","cocacola","dhl","texaco",
            "fosters","fedex","aldi","chimay","shell","becks","tsingtao","ford","carlsberg","bmw","pepsi","esso","heineken",
            "erdinger","corona","milka","ferrari","nvidia","rittersport","ups","starbucks"]
def getNum():
    f = open('a.txt','r+')
    line = f.readline()
    num = 0
    list = []
    for i in line:
        if i == '1' or i == '0':
            list.append(int(i))
            num += 1
            if(num == 8240):
                break
        else:
            continue
    return list
def computeAp(list1):
    listTP=[]
    listFP=[]
    listPre=[]
    listRec=[]
    Ap = 0
    for i in range(7):
        end = (i+1)*10
        listTP.append(sum(list1[0:end]))
        listFP.append(end-sum(list1[0:end]))
    for i in range(7):
        if listTP[i]+listFP[i] == 0:
            listPre.append(0)
        else:
            listPre.append(listTP[i] / (listTP[i] + listFP[i]))

        listRec.append(listTP[i]/70)
    for i in range(7):
        if i==0:
            Ap+=listRec[i]*listPre[i]
        else:
            Ap += listPre[i]*(listRec[i]-listRec[i-1])
    return Ap
def getMap(list):
    listAp = []
    num = -1
    list1 = []
    for line in list:
        num += 1
        if(num == 2240):
            break
        if(num % 70 ==0):
            list1 = []
        list1.append(int(line))
        if(num%70 == 69):
            listAp.append(computeAp(list1))
    return listAp
if __name__ == "__main__":
    #print(getNum())
    listAp = getMap(getNum())
    for i in range(len(listAp)):
        print(logoList[i]+"_AP:      "+str(listAp[i]))
    print("Map:      " + str(sum(listAp) / 32))