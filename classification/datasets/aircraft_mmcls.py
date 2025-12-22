import os
import numpy as np
from mmcls.datasets import CustomDataset
from mmcls.datasets import DATASETS
# from torchvision.datasets.utils import download_url, extract_archive


@DATASETS.register_module()
class FGVCAircraftDataset(CustomDataset):
    """FGVC Aircraft Dataset in MMClassification format.

    Args:
        data_root (str): Root directory of the dataset.
        train (bool): Whether to load the train split. If False, loads the test split.
        class_type (str): One of ['variant', 'family', 'manufacturer'].
        download (bool): If True, downloads the dataset if not present.
    """

    url = 'http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz'
    class_types = ('variant', 'family', 'manufacturer')
    img_folder = os.path.join('data', 'images')

    def __init__(self, data_root, data_prefix=' ', train=True, class_type='variant', download=False, **kwargs):
        data_prefix = os.path.join(data_root, self.img_folder)

        self.split = 'trainval' if train else 'test'
        if class_type not in self.class_types:
            raise ValueError(f'class_type must be one of {self.class_types}, got {class_type}')
        self.class_type = class_type

        self.data_dir = os.path.join(data_root)
        self.img_dir = os.path.join(self.data_dir, 'data', 'images')
        self.classes_file = os.path.join(self.data_dir, 'data', f'images_{class_type}_{self.split}.txt')

        super().__init__(data_prefix=data_prefix, **kwargs)
    
    def load_annotations(self):
        image_ids = []
        targets = []
        with open(self.classes_file, 'r') as f:
            for line in f:
                split_line = line.strip().split(' ')
                image_ids.append(split_line[0])
                targets.append(' '.join(split_line[1:]))

        classes = np.unique(targets)
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        targets = [class_to_idx[c] for c in targets]

        self.CLASSES = list(classes)
        data_infos = []
        for img_id, target in zip(image_ids, targets):
            info = {'img_prefix': self.img_dir}
            info['img_info'] = {'filename': f'{img_id}.jpg'}
            info['gt_label'] = np.array(int(target), dtype=np.int64)
            data_infos.append(info)
        return data_infos