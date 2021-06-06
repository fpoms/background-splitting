import torch
import os.path
import random 

from torch.utils.data import Dataset
from typing import Dict, List, Optional

class AuxiliaryDataset(Dataset):
    def __init__(self, dataset: Dataset, 
                 foreground_categories: List[int],
                 auxiliary_labels: Optional[Dict[str, int]]=None,
                 labeled_subset: Optional[float]=None,
                 restrict_aux_labels: bool=True):
        self.base_dataset = dataset
        if auxiliary_labels is None:
            auxiliary_labels = {}
        self.aux_labels = torch.tensor(
            [auxiliary_labels.get(os.path.basename(path), -1)
             for path in self.base_dataset.paths],
            dtype=torch.long)
        self.foreground_categories = foreground_categories
        self.cat_to_label = {
            cat: idx + 1 for idx, cat in enumerate(foreground_categories)}

        base_len = len(self.base_dataset)
        if labeled_subset:
            self.labeled_subset = \
                random.sample(list(range(base_len)),
                              int(base_len * labeled_subset))
        else:
            self.labeled_subset = list(range(base_len))
        self.restrict_aux_labels = restrict_aux_labels

    def __len__(self):
        if self.restrict_aux_labels:
            return len(self.labeled_subset)
        else:
            return len(self.base_dataset)

    def __getitem__(self, iindex):
        index = self.labeled_subset[iindex] if self.restrict_aux_labels else iindex
        image, main_label = self.base_dataset[index]
        if index not in self.labeled_subset:
            # Not labeled
            main_label = -1
        elif main_label not in self.cat_to_label:
            # Background label
            main_label = 0
        else:
            # Foreground label
            main_label = self.cat_to_label[main_label]
        aux_label = self.aux_labels[index]

        return image, main_label, aux_label

