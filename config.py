import warnings
import torch as t

class DefaultConfig():
    env = 'dafult' # visdom环境
    vis_port = 8097 # visdom端口

    data_root = './dataset/FlickrLogos-32_dataset_v2/FlickrLogos-v2' # 图片存放路径
    load_model_path = 'C:\\Users\\19575\\Desktop\\study\\bishe\\code1\\queryLogo\\checkpoints\\599_localize_v8.pth' # 加载预训练的模型的路径

    batch_size = 2
    use_gpu = True
    num_workers = 4
    plot_every = 2 # 每10个可视化一次
    save_model_epoch = 10 # 每100个epoch保存一下模型
    pos_weight =  t.Tensor([3]*256)

    epoches  = 20000
    lr = 0.0004 # initial lr
    momentum = 0.9
    weight_decay = 0.0005
    gaussian_ini_std = 0.01
    mask_image_path = "./misc/a.png"
    box_image_path = "./misc/b.png"
    #basepath = "C:/Users/19575/Desktop/study/bishe/code1/queryLogo"
    query_image_path = "./dataset/FlickrLogos-32_dataset_v2/FlickrLogos-v2/classes/crop/apple/3008903118.jpg"
    target_image_path = "./dataset/FlickrLogos-32_dataset_v2/FlickrLogos-v2/classes/jpg/apple/3008903118.jpg"
    #query_image_path =  "D:/PIC/other/cat0.png"
    #target_image_path = "D:/PIC/other/cat1.png"
    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        opt.device = t.device('cuda') if opt.use_gpu else t.device('cpu')

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))

opt = DefaultConfig()