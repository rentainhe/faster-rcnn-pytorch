## model
build faster-rcnn model

## dict structure
- [__utils.bbox_tools__](https://github.com/rentainhe/faster-rcnn-pytorch/blob/master/model/utils/bbox_tools.py)
- [__utils.creator_tools__]()

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