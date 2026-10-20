import torch
from torchvision.datasets import CIFAR100
from PIL import Image
import numpy as np
from .utils_sampler import superclass, make_imb_data

class CustomCIFAR100(CIFAR100):
    def __init__(self, class_num, max_num, imb_ratio, **kwds):
        super().__init__(**kwds)
        self.targets = superclass(self.targets)
        n_per_class = make_imb_data(max_num, class_num, imb_ratio, 'long')
        class_indices = {}
        for idx, label in enumerate(self.targets):
            class_indices.setdefault(label, []).append(idx)
        sampled_indices = []
        for class_idx, target_num in enumerate(n_per_class):
            indices = class_indices.get(class_idx, [])
            sample_size = min(target_num, len(indices))
            if sample_size > 0:
                chosen = np.random.choice(indices, sample_size, replace=False)
                sampled_indices.extend(chosen.tolist())
        self.data = self.data[sampled_indices]
        self.targets = np.array(self.targets)[sampled_indices]
        class_num = len(np.unique(self.targets))
        self.idxsPerClass = [np.where(np.array(self.targets) == idx)[0] for idx in range(class_num)]
        self.idxsNumPerClass = [len(idxs) for idxs in self.idxsPerClass]
        print(self.idxsNumPerClass)

    def __getitem__(self, idx):
        img = self.data[idx]
        img = Image.fromarray(img).convert('RGB')
        imgs = self.transform(img)
        target = self.targets[idx]
        return imgs, target, idx