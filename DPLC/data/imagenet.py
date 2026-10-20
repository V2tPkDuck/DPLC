import torch
import numpy as np
import os
import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import IMG_EXTENSIONS
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from datasets import load_dataset, DownloadConfig
from .utils_sampler import make_imb_data, _sample_indices_from_class_indices
import numpy as np
# Import the robust sampling functions from your first script
# Adjust the import path if longtail_sampler is in a different directory
from .utils_sampler import make_imb_data, _sample_indices_from_class_indices

class CustomTinyImagenetLT(ImageFolder):
    def __init__(self, class_num, max_num, imb_ratio, **kwds):
        super().__init__(**kwds)
        
        # 1. Generate the target distribution using the clean mathematical formulation
        num_per_class = make_imb_data(max_num, class_num, imb_ratio)
        
        # 2. Group original sample indices by class
        targets = np.asarray([s[1] for s in self.samples])
        class_indices = {}
        for idx, target in enumerate(targets):
            if target not in class_indices:
                class_indices[target] = []
            class_indices[target].append(idx)
            
        # 3. Apply the standardized sampling logic
        sampled_indices = _sample_indices_from_class_indices(class_indices, num_per_class)
        
        # 4. Overwrite samples with the long-tail subset
        self.samples = [self.samples[idx] for idx in sampled_indices]
        self.targets = [s[1] for s in self.samples]
        self.img_num = len(self.samples)
        # print("Num per class:", num_per_class)
        # print("Non-zero classes:", np.count_nonzero(num_per_class))
        # print("Total samples:", np.sum(num_per_class))
        # print(f"Tiny ImageNet LT initialized with {self.img_num} images.")

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = self.loader(path)
        
        if self.transform is not None:
            images = self.transform(img)
        else:
            images = img
            
        # Crucial: Returning idx aligns with the subset length, 
        # allowing your mini-cluster assignments to map correctly.
        return images, target, idx

class CustomImageNet50LT(ImageFolder):
    """
    Local-only ImageNet-50 dataset with controllable long-tail distribution.
    Root must contain class subfolders (e.g., .../train , .../val).
    The test set is also subsampled according to imb_ratio, satisfying the
    requirement of "test-set long-tail clipping".
    """
    def __init__(self, root, class_num, max_num, imb_ratio, transform=None):
        super().__init__(root=root, transform=transform)

        # Apply long-tail subsampling
        num_per_class = make_imb_data(max_num, class_num, imb_ratio)
        targets = np.asarray([s[1] for s in self.samples])
        class_indices = {}
        for idx, target in enumerate(targets):
            if target not in class_indices:
                class_indices[target] = []
            class_indices[target].append(idx)

        sampled_indices = _sample_indices_from_class_indices(class_indices, num_per_class)
        self.samples = [self.samples[idx] for idx in sampled_indices]
        self.targets = [s[1] for s in self.samples]
        self.img_num = len(self.samples)
        # print(f"ImageNet50-LT initialized: {self.img_num} images from {root}")

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = self.loader(path)
        if self.transform is not None:
            images = self.transform(img)
        else:
            images = img
        return images, target, idx

class CustomImageNet100LT(ImageFolder):
    EXTENSIONS = IMG_EXTENSIONS
    def __init__(self, root, class_num, max_num, imb_ratio, transform=None):
        super().__init__(root=root, transform=transform)
        num_per_class = make_imb_data(max_num, class_num, imb_ratio)
        targets = np.asarray([s[1] for s in self.samples])
        class_indices = {}
        for idx, target in enumerate(targets):
            class_indices.setdefault(target, []).append(idx)
        sampled_indices = _sample_indices_from_class_indices(class_indices, num_per_class)
        self.samples = [self.samples[idx] for idx in sampled_indices]
        self.targets = [s[1] for s in self.samples]
        self.img_num = len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = self.loader(path)
        if self.transform is not None:
            images = self.transform(img)
        else:
            images = img
        return images, target, idx
    
class CustomImageNetLT(Dataset):
    num_classes = 1000

    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        self.targets = self.labels
        self.class_data = [[] for i in range(self.num_classes)]
        for i in range(len(self.labels)):
            y = self.labels[i]
            self.class_data[y].append(i)

        self.cls_num_list = [len(self.class_data[i]) for i in range(self.num_classes)]
        print(self.cls_num_list)
        self.transform = transform
        print(len(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path1 = self.img_path[index]
        label = self.labels[index]
        with open(path1, 'rb') as f:
            img = Image.open(f).convert('RGB')
        images = self.transform(img)
        return images, label, index