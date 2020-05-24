import torch as t
import torch.nn as nn
from torch.utils.data import DataLoader
from torchnet import meter

from data import FLogo
from config import opt
from models.QueryLogo import Net
from utils import Visualizer

import tqdm
import numpy as np
from compute import Detection


def train(**kwargs):
    opt._parse(kwargs)
    vis = Visualizer(opt.env,port = opt.vis_port)
    device = t.device('cuda') if opt.use_gpu else t.device('cpu')

    # 数据加载
    train_data = FLogo(opt.data_root,train=True)
    train_dataloader = DataLoader(train_data,opt.batch_size,shuffle=True,num_workers=opt.num_workers)

    '''
    # 以下内容是可视化dataloader的数据的
    一 检查dataset是否合理
    二 为了写论文凑图
    
    dataiter = iter(train_dataloader)
    img1,img2,lable=dataiter.next()
    img1 = tv.utils.make_grid((img1+1)/2,nrow=6,padding=2).numpy()
    img2 = tv.utils.make_grid((img2+1)/2,nrow=6,padding=2).numpy()
    plt.figure()
    plt.imshow(np.transpose(img1, (1, 2, 0)))
    plt.figure()
    plt.imshow(np.transpose(img2, (1, 2, 0)))
    plt.figure()
    lables = label.unsqueeze(1)  # lables
    mask = tv.utils.make_grid(lables,nrow=6,padding=2).numpy()
    plt.imshow(np.transpose(mask, (1, 2, 0)))
    plt.show()


from torchvision.transforms import ToPILImage
import numpy as np
import matplotlib.pylab as plt
train()
    '''

    # 网络
    net = Net()
    net.train()

    # 加载预训练模型
    if opt.load_model_path:
        net.load_state_dict(t.load(opt.load_model_path,map_location = lambda storage,loc:storage),False)
        print('已加载完。。')
    else:
        # 模型初始化
        for m in net.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                print('模型参数完成初始化。。')
    net.to(device)

    # 损失函数和优化器
    criterion = nn.BCEWithLogitsLoss(pos_weight=opt.pos_weight.to(device))
    optimizer = t.optim.SGD(net.parameters(),lr=opt.lr, momentum=opt.momentum,weight_decay=opt.weight_decay)

    # 使用meter模块
    loss_meter = meter.AverageValueMeter()

    # 学习率调整策略
    # scheduler = StepLR(optimizer, step_size=1000, gamma=0.5)

    for epoch in range(opt.epoches):
        loss_meter.reset() # 重置loss_meter??
        for ii,(target_img,query_logo,mask) in tqdm.tqdm(enumerate(train_dataloader)):
            print(target_img.shape)
            # 训练
            target_img = target_img.to(device)
            query_logo = query_logo.to(device)

            mask = mask.to(device)

            optimizer.zero_grad()

            output = net(query_logo,target_img)
            output = output.squeeze()
            predict = t.sigmoid(output)
            # predict_mask = t.sigmoid(output) # true output should be sigmoid
            # ipdb.set_trace()
            true_mask = mask/255

            # predict = output.view(output.size(0),-1)
            # target = true_mask.view(true_mask.size(0),-1)
            # ipdb.set_trace()
            # print(predict.size(),target.size())


            # loss = criterion(F.softmax(output,dim=2),true_mask)
            loss = criterion(output,true_mask)
            # print(loss.item())

            loss.backward()
            optimizer.step()

            # meter update and visualize
            loss_meter.add(loss.item())
            if (ii+1)%opt.plot_every == 0:

                vis.img('target_img', ((target_img + 1) / 2).data[0])
                vis.img('query_logo', ((query_logo + 1) / 2).data[0])
                vis.img('truth groud', (true_mask.data[0]))
                vis.img('predict', predict.data[0])
                pre_judgement = predict.data[0]
                pre_judgement[pre_judgement > 0.5] = 1  # 改成0.7怎么样！
                pre_judgement[pre_judgement <= 0.5] = 0
                vis.img('pre_judge(>0.5)', pre_judgement)

                # vis.img('pre_judge', pre_judgement)
                # vis.log({'predicted':output.data[0].cpu().numpy()})
                # vis.log({'truth groud':true_mask.data[0].cpu().numpy()})

        print('finish epoch:',epoch)
        # vis.log({'predicted':output.data[0].cpu().numpy()})
        vis.plot('loss',loss_meter.value()[0])

        if (epoch+1) %opt.save_model_epoch == 0:
            vis.save([opt.env])
            t.save(net.state_dict(),'checkpoints/%s_localize_v6.pth' % epoch)

        # scheduler.step() # 更新学习率
#
if __name__ == '__main__':
    print(Detection.test())
    np.set_printoptions(threshold=np.inf)
    #import fire
    #fire.Fire()
    #train()






