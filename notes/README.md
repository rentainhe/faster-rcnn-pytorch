# basic knowledge
some basic knowledge of object detection

- [目标检测中mAP的计算方法](https://zhuanlan.zhihu.com/p/94597205)

## mAP
### basic metric
- __TP(True Positive): 一个正确的检测, 检测的IOU > threshold. 即预测的边界框中分类正确并且边界框坐标正确的数量.__
- __FP(False Positive): 一个错误的检测, 检测的IOU < threshold. 即预测的边界框中分类错误或者边界框坐标不达标的数量, 即预测出的所有边界框中除去正确的边界框, 剩下的边界框数量.__
- __FN(False Negative): 一个没有被检测出来的ground truth. 所有没有预测到的边界框的数量, 即正确的边界框(ground truth)中除去被预测正确的边界框，剩下的边界框的数量.__
- __Precision(准确率 / 精确率): 准确率是模型只找到相关目标的能力, 等于 TP / (TP + FP). 即模型给出的所有预测结果中命中真实目标的比例.__
- __Recall(召回率), 召回率是模型找到所有相关目标的能力, 等于 TP / (TP + FN). 即模型给出的预测结果最多能覆盖多少真实目标.__
- __score / confidence: 每个预测边界框的分数/置信度, 模型需要输出每个边界框的分类结果及坐标的同时, 还需要输出这个边界框包含物体的可能性.__
- __PR曲线(Precision Recall curve)__


一般对于多分类目标检测任务, 会分别计算每个类别的TP, FP, FN数量, 进一步计算每一个类别的Precision, Recall
### example
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
```