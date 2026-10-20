from torchvision.datasets import STL10
from PIL import Image
import numpy as np
from .utils_sampler import make_imb_data, _sample_indices_from_class_indices

class CustomSTL10(STL10):
    def __init__(self, class_num, max_num, imb_ratio, **kwds):
        super().__init__(**kwds)
        data, labels = self.concatenateData()
        self.data = data
        self.labels = labels
        n_per_class = make_imb_data(max_num, class_num, imb_ratio)

        class_indices = {c: [] for c in range(class_num)}
        for idx, label in enumerate(self.labels):
            class_indices[label].append(idx)

        train_idx = _sample_indices_from_class_indices(class_indices, n_per_class)

        if train_idx is not None:
            self.data = self.data[train_idx, :, :, :]
            self.labels = np.array(self.labels)[train_idx]
        self.idxsPerClass = [np.where(np.array(self.labels) == idx)[0] for idx in range(class_num)]
        self.idxsNumPerClass = [len(idxs) for idxs in self.idxsPerClass]
        print(self.idxsNumPerClass)

    def concatenateData(self):
        train_dataset = STL10(root=self.root, split='train')
        test_dataset = STL10(root=self.root, split='test')
        train_dataset.data = np.concatenate([train_dataset.data, test_dataset.data], axis=0)
        train_dataset.labels = np.concatenate([train_dataset.labels, test_dataset.labels], axis=0)
        return train_dataset.data, train_dataset.labels

    def __getitem__(self, idx):
        img = self.data[idx]
        img = Image.fromarray(np.transpose(img, (1, 2, 0))).convert('RGB')
        imgs = self.transform(img) if self.transform is not None else img
        label = self.labels[idx]
        return imgs, label, idx