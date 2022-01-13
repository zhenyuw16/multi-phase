from .builder import DATASETS
from .custom import CustomDataset

@DATASETS.register_module()
class VOCDataset2(CustomDataset):

    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')

    def __init__(self, **kwargs):
        super(VOCDataset2, self).__init__(**kwargs)
        #print(self.img_prefix)
        #'''

