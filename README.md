# Doc: TransPillars

## PointPillars

本文使用PointPillars作为基本模型，PointPillars与其他基于体素的模型不同，仅在x-y平面以固定大小将点云进行离散化，以此方法生成柱体代替立方体体素。每个体素内的点用来生成特征向量，然后将获得的柱状特征散射回它们在场景中的相应位置以形成伪图像表示。然后伪图像特征经过图像金字塔处理，最后下采样的特征会被逆卷积上采样，基于连接后的特征，使用检测头生成最终的检测结果。


<img src="figs/transpillars.png" alt="" style="zoom:80%;" />

在特征提取部分，给定一个点云输入序列 ![](https://latex.codecogs.com/svg.image?%5C%7BI_%7BT-n%7D%5C%7D%5E%7BN-1%7D_%7Bn=0%7D)，一个公用的特征提取器用来提取多尺度特征 ![](https://latex.codecogs.com/svg.image?%5C%7B%5Cmathbf%7BF%7D_%7BT-n%7D%5C%7D%5E%7BN-1%7D_%7Bn=0%7D)。后续的特征聚合模块会聚合过往帧的有用信息来丰富当前帧的特征表示。

## TransPillars

为了实现体素级别的的聚合，面临着两个挑战。第一个是由注意力机制和全局匹配导致的高额的计算开销以及内存占用，为了缓解这个问题本文采用了最近提出的“可变形注意力”（deformable attention）的一个变体。第二个挑战是捕捉快速移动物体的运动以建立跨帧对应关系，由于小体素尺寸带来的大搜索空间，很难直接在精细特征上实现。为了解决这个问题，本文利用基本模型生成的多尺度特征设计了一种新颖的从粗到细的聚合策略。 具体来说，本文采用基于transformer的融合聚合模块（Fusion Aggregation Module）从粗略的特征图开始执行特征融合和特征聚合。然后将输出的聚合特征与下一个尺度级别的特征图融合以进行后续聚合。最后所有FAM的输出会像PointPillars中一样组合到一起来生成最终的预测。

## Fusion Aggregation Module

<img src="figs/trans.png" alt="" style="zoom:60%;" />

FAM包括特征
