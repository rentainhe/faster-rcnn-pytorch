import os
import xml.etree.ElementTree as ET
import numpy as np
from .util import read_image

# VOC 数据集各个 LABEL 对应的名称
VOC_BBOX_LABEL_NAMES = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor')

class VOCBboxDataset:
    def __init__(self, data_dir, split='trainval', use_difficult=False, return_difficult=False):
        id_list_file = os.path.join(
            data_dir, 'ImageSets/Main/{0}.txt'.format(split)
        )

        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.data_dir = data_dir
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        self.label_names = VOC_BBOX_LABEL_NAMES

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        """Return the i-th example.

        Return a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB

        Args:
            i (int): The index of the example

        Returns:
            tuple of an image and bounding boxes
        """
        id_ = self.ids[i]
        anno = ET.parse(
            os.path.join(self.data_dir, 'Annotations', id_ + 'xml')
        )
        bbox = list()
        label = list()
        difficult = list()
        for obj in anno.findall('object'):
            # when not using difficult split, and the object is difficult, skip it.
            if not self.use_difficult and int(obj.find('difficult').text) == 1:
                continue

            difficult.append(int(obj.find('difficult').text))
            bbox_anno = obj.find('bndbox')
            # subtract 1 to make pixel indexes 0-based
            bbox.append([
                int(bbox_anno.find(tag).text) - 1
                for tag in ('ymin', 'xmin', 'ymax', 'xmax')
            ])
            name = obj.find('name').text.lower().strip()
            label.append(VOC_BBOX_LABEL_NAMES.index(name))
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        # When `use_difficult==False`, all elements in `difficult` are False.
        difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)  # PyTorch don't support np.bool