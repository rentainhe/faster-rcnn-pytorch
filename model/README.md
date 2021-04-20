# faster-rcnn-pytorch
build faster rcnn on pytorch from scratch

## project structure documents
- [__utils: proposal creator__](https://github.com/rentainhe/faster-rcnn-pytorch/tree/master/model/utils)
- [__region_proposal_network.py__](https://github.com/rentainhe/faster-rcnn-pytorch/tree/master/model/region_proposal_network.py)

## contents
### Region Proposal Network
Region Proposal Network introduced in Faster R-CNN

__init:__
- __in_channels=512: 经过VGG或者Darknet提取后的特征__
- __mid_channels=512: 将特征进行再一次的卷积后的维度__
- __ratios=[0.5, 1, 2]: 用于生成anchor的参数__
- __anchor_scales=[8, 16, 32]: 用于生成anchor的参数__
- __feat_stride=16: 经过特征提取后feature map相比于输入进来的原图的放缩倍数, 这里表示原图大小是feature map的16倍__
- __proposal_creator_params=dict(): proposal creator的参数__

__forward:__
- __x: 经过特征提取后的 feature map__
- __img_size: 原图大小__
- __scale=1.: 候选框的 scale 参数__

__returns:__
- __rpn_locs: Predicted bounding box offsets and scales for anchors. Its shape is (N, H * W * A, 4), "A" means number of anchors assigned to each pixel__
- __rpn_scores: Predicted foregound scores for anchors__
- __rois: A bounding box array containing coordinates of proposal boxes. This is a concatenation of bounding box arrays from multiple images in the batch.__
- __roi_indices: An array containing indices of images to  which RoIs correspond to. 记录每个图片对应的index, Its shape is (R, )__
- __anchor: Coordinates of enumerated shifted anchors. Its shape is (H * W * A, 4)__


具体流程:
- 保存 feature map 的长宽
- 通过 `_enumerate_shifted_anchor` 得到一张图上所有 grid 对应的 anchor, 因为原始的候选anchor只是候选框大小, 并没有根据图像的位置进行调整
- 获得 n_anchor 数量, 表示每个 grid 所对应的 anchor 数量
- 第一步: 对输入进来的 feature map 进行一次再卷积操作
- 第二步: 经过一层 1*1 的卷积, mid_channels -> anchor * 4, 表示对每个 grid 中的每个 anchor 都预测一个 predicted bbox offsets and scales
- 第三步: 经过一层 1*1 的卷积, mid_channels -> anchor * 2, 表示对每个 grid 中的每个 anchor 都预测一个前景概率
- 最后将预测出来的前景分数, 偏移值, bbox, 传入`ProposalCreator`中去, 得到我们的 RPN 最终的候选框




