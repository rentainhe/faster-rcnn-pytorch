## model
build faster-rcnn model

## dict structure
- [__bbox_tools__](https://github.com/rentainhe/faster-rcnn-pytorch/blob/master/model/utils/bbox_tools.py)
- [__creator_tools__](https://github.com/rentainhe/faster-rcnn-pytorch/blob/master/model/utils/creator_tool.py)

## contents
### 1. utils.bbox_tools

- __loc2bbox (src_bbox, loc): Decode bounding boxes from bounding box offsets and scales, 根据宽高比率和中心点偏差来调整bounding box位置__
  - __src_bbox: source bounding box__
  
  - __loc: bounding box offsets and scales__
  
- __bbox2loc (src_bbox, dst_bbox): Encodes the source and the destination bounding boxes to "loc", 根据源bbox和目标bbox来计算中心点偏差以及宽高对应比率__
  - __src_bbox: source bounding box__
  
  - __dst_bbox: destination bounding box__

- __bbox_iou (bbox_a, bbox_b): Calculate the Intersection of Unions (IoUs) between bounding boxes.__
  - __根据两组bounding box来计算两两对应的 IoU__
  
- __generate_anchor_base (base_size=16, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]): Generate anchor base windows by enumerating aspect ratio and scales__
  - __根据对应的比率和anchor size来生成候选框__
 
 
### 2. utils.creator_tools
#### ProposalTargetCreator
Assign ground truth bounding boxes to given RoIs, 为传入进来的 RoI 分配一个 ground truth 类别

所需的参数和返回值如下所示:

__init:__
- __n_sample=128: 采样数量__
- __pos_ratio=0.25: 所采的positive roi占比__
- __pos_iou_thresh=0.5: IoU threshold for a RoI to be considered as a foreground__
- __neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.0: RoI is considered to be the background if IoU is in [hi, lo]__

__call:__
- __roi (R, 4): Region of Interests (RoIs) from which we sample.__
- __bbox (R, 4): The coordinates of ground truth bounding boxes.__
- __label (R, ): Ground truth bounding box labels.__
- __loc_normalize_mean: 对 Offset 和 Scales 的均值归一化__
- __loc_normalize_std__

__returns:__
- __sample_roi (S, 4): 采样的 Region of Interests__
- __gt_roi_loc (S, 4): 所采样的 Region of Interests 相对于其分配的 Ground truth 的 Offsets 和 Scales__
- __gt_roi_label (S, ): 所采样的 Region of Interests 所对应的 Label, 如果被分配到 0 的话则为 background__

具体流程如下:

- 将传入的 roi 和 bbox 拼接到一起, 这一步是为了在训练的时候能考虑到所有的 ground truth, 至少在训练中得包含 ground truth
- 对拼接后的 roi 和 bbox 两两之间计算 ious, 使用 `bbox_iou` 函数
- 为每一个 roi 分配一个具体的 ground truth 类别, 然后根据 pos_iou_thresh 来选取对应数量的 positive roi 作为正样本
- 根据 neg_iou_thresh_hi 和 neg_iou_thresh_lo 选取对应的负样本
- 计算所采样的 roi, 相对于 ground truth 的 bounding box 的中心点偏移值以及长宽比率(loc), 使用 `bbox2loc` 函数
- 对偏移值进行归一化
- 返回: `所采样的 roi`, `采样的roi相对于bbox的loc`, `采样的roi的label`

#### AnchorTargetCreator
Assign the ground truth bounding boxes to anchors.

Assign the ground truth bounding boxes to anchors for training `Region Proposal Networks`.

采样训练 RPN 网络

__init:__
- __n_sample=256: 总共采样的数量__
- __pos_iou_thresh=0.7: 选取作为前景(positive)样本的 threshold__
- __neg_iou_thresh=0.3: 选取作为背景(negative)样本的 threshold__
- __pos_ratio=0.5: 前景样本的比例__

__call:__
- __bbox (R, 4): Coordinates of ground truth bounding boxes__
- __anchor (S, 4): Coordinates of anchors__
- __img_size (H, W): Image size__

__returns:__
- __loc (S, 4): Offsets and scales to match the anchors to the ground truth bounding boxes, 每个anchor与其对应的bounding box的中心点偏移值以及长宽比例__
- __label (S, ): 每个anchor对应的标签, 其中 1 = positive, 0 = negative, -1 = ignore__

具体流程如下:
- 得到图像的宽和高, 先通过内部的函数`_get_inside_index`得到位于图像内部的anchor
- 调用`_create_label`函数来获得这些anchor点对应的label
- `_create_label`函数的内部流程:
  - 先创造一个对应anchor数量的空数组, 用 -1 填充
  - 调用`_calc_ious`来获得三组index:
    - `argmax_ious`: 每个 anchor 对应的最大 iou 的 ground truth bounding box 的 index
    - `max_ious`: 每个在图片内的anchor, 所对应的最大 iou
    - `gt_argmax_ious`: 每个 ground truth bounding box 对应最大 iou 的 anchor 的 index
  - `_create_label`首先对`max_ious < neg_iou_thresh=0.3`的 anchor 赋予 label 0 
  — 然后对每个 ground truth bbox 对应最大 iou 的 anchor 赋予 label 1
  - 然后再对 `max_ious > pos_iou_thresh=0.7` 的 anchor 赋予 label 1
  - 然后再随机选取 256 个 anchor, 包含128个positive anchor 和 128个negative anchor, 对其他的 anchor 点赋予 label -1 表示 ignore 忽略不计
  - 然后返回`argmax_ious` 和 `label` 数组
- 根据 `argmax_ious` 来计算每个 anchor 相对于自己可能的 ground truth bbox 的中心点偏移值和长宽比率, 因为这是我们的优化目标
- 返回对应的 label 和 优化目标loc

#### ProposalCreator
Proposal regions are generated by calling this object

__init:__
- __parent_model: 通过 parent_model.training 来判断模型是处于训练阶段还是测试阶段, 因为这两个阶段采样数量存在差异__
- __nms_thresh=0.7: 采样为前景的 nms threshold__
- __n_train_pre_nms=12000: 训练阶段经过 nms 前的 roi 数量__
- __n_train_post_nms=2000: 经过 nms 后所剩余的 roi 数量__
- __n_test_pre_nms=6000: 测试阶段经过 nms 前的 roi 数量__
- __n_test_post_nms=300: 经过 nms 后所剩余的 roi 数量__
- __min_size=16: 最小的候选框大小, 所选候选框不能小于这个大小乘以 scale 参数__

__call:__
- __loc: Predicted offsets and scaling to anchors. Its shape is (R, 4)__
- __score: Predicted foreground probability for anchors. Its shape is (R, )__
- __anchor: Coordinates of anchors. Its shape is (R, 4)__
- __img_size: A tuple which contains image size after scaling__
- __scale: The scaling factor used to scale an image after reading it from a file__ 

__returns:__
- __roi: An array of coordinates of proposal boxes. Its shape is  (S, 4).

具体流程如下:
- 首先根据传进来的`parent_model.training`来判断模型是处于训练阶段还是测试阶段
- 然后再将传进来预测的 `loc` 转化为对应的 predicted bbox , 因为 `loc` 是我们 `RPN` 预测出来的
- 然后对预测的 bbox 进行 clip 操作, 将 predicted bbox 限制在图像范围内, 具体做法就是超过图像的部分裁剪到图像大小, 使用 `numpy.clip` 函数
- 根据我设置的 `min_size` 参数, 裁减掉过于小的 predicted bbox
- 根据传入进来的 predicted foreground probability 进行排序操作, 选取 `n_pre_nms` 个候选框进行 nms 操作, `n_pre_nms` 是由 `n_train_pre_nms` 等决定的
- 将这些保留下来的框送入 nms 进行处理, 保留最终的结果, 数量为 `n_post_nms` 个


