import os
import scipy.io
from os.path import join
from mmcls.datasets import CustomDataset
from mmcls.datasets import DATASETS
import numpy as np


@DATASETS.register_module()
class StanfordDogsDataset(CustomDataset):
    """Stanford Dogs Dataset for MMClassification."""

    def __init__(self, data_root, data_prefix=' ', train=True, download=False, **kwargs):
        self.train = train
        self.images_folder = join(data_root, 'Images')
        self.annotations_folder = join(data_root, 'Annotation')
        self.split_file = 'train_list.mat' if train else 'test_list.mat'
        self.data_root = data_root

        self.class_names = sorted(os.listdir(self.images_folder))
        self.CLASSES = self.class_names  # required for MMCls

        super().__init__(data_prefix=self.images_folder, **kwargs)
    
    ### override load_annotations func
    def load_annotations(self):
        data_infos = []
        mat = scipy.io.loadmat(join(self.data_root, self.split_file))
        image_list = [item[0][0] for item in mat['annotation_list']]
        label_list = [int(item[0]) - 1 for item in mat['labels']]  # to 0-based
        for img_name, label in zip(image_list, label_list):
            info = {'img_prefix': os.path.join(self.data_root, "Images")}
            info['img_info'] = {'filename': str(img_name)+".jpg"}
            info['gt_label'] = np.array(int(label))
            data_infos.append(info)
        return data_infos
