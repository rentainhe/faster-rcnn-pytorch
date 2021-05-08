# basic knowledge
some basic knowledge of object detection

- [目标检测中mAP的计算方法](https://zhuanlan.zhihu.com/p/94597205)
- [目标检测的指标AP和mAP](https://zhuanlan.zhihu.com/p/140062567)

## mAP
### basic metric
- __TP(True Positive): 一个正确的检测, 检测的IOU > threshold. 即预测的边界框中分类正确并且边界框坐标正确的数量.__
- __FP(False Positive): 一个错误的检测, 检测的IOU < threshold. 即预测的边界框中分类错误或者边界框坐标不达标的数量, 即预测出的所有边界框中除去正确的边界框, 剩下的边界框数量.__
- __FN(False Negative): 一个没有被检测出来的ground truth. 所有没有预测到的边界框的数量, 即正确的边界框(ground truth)中除去被预测正确的边界框，剩下的边界框的数量.__
- __Precision(准确率 / 精确率): 准确率是模型只找到相关目标的能力, 等于 TP / (TP + FP). 即模型给出的所有预测结果中命中真实目标的比例.__
- __Recall(召回率), 召回率是模型找到所有相关目标的能力, 等于 TP / (TP + FN). 即模型给出的预测结果最多能覆盖多少真实目标.__
- __score / confidence: 每个预测边界框的分数/置信度, 模型需要输出每个边界框的分类结果及坐标的同时, 还需要输出这个边界框包含物体的可能性.__
- __PR曲线(Precision Recall curve)__
```
为什么需要Precision和Recall?
因为Precision表示的是预测为正确的pred_bbox中, 有多少是和ground truth匹配的. 
但是如果有100个样本, 预测了一个正例, 而且这个正例是正确的, 那么准确率为100%. 
所以需要Recall来平衡, Recall衡量的是找到所有目标的能力
```

一般对于多分类目标检测任务, 会分别计算每个类别的TP, FP, FN数量, 进一步计算每一个类别的Precision, Recall
### example
#### 1. TP、FP和FN的计算
对于模型训练而言, 测试阶段时会组成一个batch输入, 这时计算TP, FP, FN时就会以batch为单位计算

- 计算TP: 对于一张图片中某个类别的全部预测框, 如果一个pred_bbox与ground truth的IOU大于阈值, 则认为这个框预测正确, 计作一个TP. 如果出现多个预测框与同一个ground truth的IOU都大于阈值, 这时通常只将预测框中score最大的算作TP, 其余算作FP
- 计算FP: 对于一张图片中某个类别的所有预测的边界框, 除了TP之外的记作FP, 在实际应用中会有一个TP标志位, 如果当前预测框预测正确, 那么这个标志位设置为1代表TP, 否则为0代表FP
- 计算FN: 对于一张图片中某个类别的所有ground truth边界框, 除了TP之外的记为FN

```
例如一张图片中有100个预测的边界框, 其中包括A类别30个, B类别30个, C类别40个. 
如果包含A类别的gt有4个, 被正确预测到了2个
A类别: TP+2, FP+28, FN+2
如果这个图片中没有B和C类别的gt
B类别: TP+0, FP+30, FN+0
C类别: TP+0, FP+40, FN+0
如果图片中包含一个D类别的gt
D类别: TP+0, FP+0, FN+1
对测试集中的所有batch中的所有图片都按照上述方法计算TP、FP、FN并累加, 最终得到A,B,C,D,...所有类别在当前数据集中所有图片的TP、FP和FN
```
#### 2. PR曲线的计算
根据1中的例子, 对于某个类别A, 我们计算每张图片中A类别TP和FP的数量并进行累加, 即可得到类别A在整个数据集中的TP和FP的数量, 通过计算 `TP / (TP + FP)` 即可得到类别A的`Precision`, 
```
Precision = TP / (TP + FP), 是受整个模型输出框所影响的, 如果输出框数量发生变化, 则Precision的值会发生变化
```
```
Recall = TP / ground truth, 其中ground truth为某个类别的物体在所有图片中的ground truth数量之和. 
如果输出的预测框数量变化, 那么TP数量会变化, Recall值就会变化
```

```
假设一个数据集中共有5张图片
1. 第一张图片中有2个A类别的ground truth, 有3个A类别的预测框, score分别为(0.3, 0.5, 0.9).
按照上述计算TP的方法(由score从大到小顺序匹配), 发现score为0.3和0.9的与gt匹配, 则将这两个记为TP. 
并在这一步建立用于计算PR曲线的数组metric和记录A类别gt总数的变量ngt
向数组中添加(0.3, 1)、(0.5, 0)和(0.9, 1)三组数据(每组数据的第一个代表预测框的score, 第二个表示预测框是否为TP), 并且此时ngt累积到3
2. 第二张图片中没有A类别的物体, 即 gt = 0, ngt += 0, 但有一个关于A类别的预测框, score = 0.45, 则向metric中加入(0.45, 0).
3. 第三张图片中有一个
```
