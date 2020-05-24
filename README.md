程序需要实现以下功能：
1. 模型定义 （完成于2020.2.23）
2. 数据处理与加载  (FlickrLogos-32 dataset.百度网盘)
3. 训练模型（Train & Validata)
4. 训练过程的可视化
5. 测试(Test & Inference)


程序文件的组织结构
```
checkpoints/
data/
    __init__.py
    dataset.py
    get_data.sh
models/
    __init__.py
    QueryBase.py
    FasterRCNN.py
utils/
****    __init__.py
    visualize.py
config.py
main.py
requirements.txt
README.md
```

**数据处理**
1. 从target image中按照mask 裁剪出logo区域 （crop.ipynb)
2. 从all.relpath.txt中删除了所有no-logo的路径
3. 暂时没有数据增强的处理。因为不会







checkpoints:

    v4和v5是vgg16（在99_localoze_v2预训练下）去掉后两个maxpool层的，就是输出形状是4x4
    v3是在v2基础上
    v2是在49_localoze
    v3-v4都是F.tanh(d_s5)
  
    v6是：vgg16去掉两个 最大池化。最后一层用sigmoid替代relu激活。数据集扩充了负样本，避免只检测图片中他认识的logo，conditionalbranch就没得意义
    