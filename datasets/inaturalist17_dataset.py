import json
import os.path
import imageio
import torch

from torch.utils.data import Dataset
from typing import Dict, List

class iNaturalist17Dataset(Dataset):
    def __init__(self, root: str, ann_path: str, transform=None):
        self.transform = transform

        with open(ann_path, 'r') as f:
            data = json.load(f)

        self.paths = []
        self.labels = []
        for image_info, ann_info in zip(data['images'], data['annotations']):
            assert image_info['id'] == ann_info['image_id']
            image_path = image_info['file_name']
            label = ann_info['category_id']
            self.paths.append(os.path.join(root, image_path))
            self.labels.append(label)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        label = self.labels[index]
        try:
            image = torch.from_numpy(
                imageio.imread(path, as_gray=False, pilmode="RGB").copy())
            image = image.permute(2, 1, 0)
        except RuntimeError as e:
            print(path)
            print(e)
            raise e
        image = self.transform(image) if self.transform else image
        return image, label
