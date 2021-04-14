## data
data preprocess

## data structure
- [__PASCAL VOC数据集标注格式__](https://zhuanlan.zhihu.com/p/33405410)

## dict structure
- [__utils.py__](https://github.com/rentainhe/faster-rcnn-pytorch/blob/master/data/util.py)

## contents
### 1. utils
The default type of the bounding box is np.float32

It has the shape as `(R, 4)`, where `R` is the bounding boxes in the image.

The second axis represents attributes of the bounding box. They are `(y_{min}, x_{min}, y_{max}, x_{max})`, where the four attributes are coordinates of the __top left__ and the __bottom right__ vertices.

- __read_image (path, dtype=np.float32, color=True): Load image file from a specific path__

- __resize_bbox (bbox, in_size, out_size): Resize bounding box according to the image size__

- __flip_bbox (bbox, size, y_flip=False, x_flip=False): Flip bounding boxes accordingly__

- __crop_bbox (bbox, y_slice=None, x_slice=None, allow_outside_center=True, return_param=False): Translate bounding boxes to fit within the cropped area of an image__

- __translate_bbox (bbox, y_offset=0, x_offset=0): Translate bounding boxes__

- __random_flip (img, y_random=False, x_random=False, return_param=False, copy=False): Randomly flip an image in vertical or horizontal direction__
