import os
import pandas as pd
from mmcls.datasets import CustomDataset
from mmcls.datasets import DATASETS
import numpy as np


@DATASETS.register_module()
class Cub2011Dataset(CustomDataset):
    """CUB-200-2011 Dataset in MMClassification format."""

    def __init__(self, ann_file=None, data_root=None, train=True, data_prefix='', **kwargs):
        self.train = train
        self.base_folder = 'CUB_200_2011/images'
        self.meta_root = os.path.join(data_root, 'CUB_200_2011')
        self.data_root = data_root

        super().__init__(data_prefix=os.path.join(self.data_root, self.base_folder), **kwargs)

        # Get class names
        class_names = pd.read_csv(os.path.join(self.meta_root, 'classes.txt'),
                                  sep=' ', names=['class_id', 'class_name'])
        self.CLASSES = class_names['class_name'].tolist()

        # Build data_list for CustomDataset
        # data_list = []
        # for _, row in self.data.iterrows():
        #     img_path = os.path.join(self.base_folder, row.filepath)
        #     label = int(row.target) - 1  # 1-based to 0-based
        #     data_list.append(dict(img_path=img_path, gt_label=label))


    ### override load_annotations func
    def load_annotations(self):
        images = pd.read_csv(os.path.join(self.meta_root, 'images.txt'),
                             sep=' ', names=['img_id', 'filepath'])
        labels = pd.read_csv(os.path.join(self.meta_root, 'image_class_labels.txt'),
                             sep=' ', names=['img_id', 'target'])
        split = pd.read_csv(os.path.join(self.meta_root, 'train_test_split.txt'),
                            sep=' ', names=['img_id', 'is_training_img'])

        df = images.merge(labels, on='img_id').merge(split, on='img_id')
        self.data = df[df.is_training_img == int(self.train)]
        print("Dataset size: ", len(self.data))

        data_infos = []
        for _, row in self.data.iterrows():
            info = {'img_prefix': os.path.join(self.data_root, self.base_folder)}
            info['img_info'] = {'filename': row.filepath}
            info['gt_label'] = np.array(int(row.target) - 1, dtype=np.int64)
            data_infos.append(info)
        return data_infos 
