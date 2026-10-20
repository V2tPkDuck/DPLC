import os

import numpy as np
import torch

from torchvision.transforms import transforms

from data.STL10 import CustomSTL10
from data.cifar10 import CustomCIFAR10
from data.cifar100 import CustomCIFAR100
from data.imagenet import CustomTinyImagenetLT, CustomImageNetLT, CustomImageNet50LT, CustomImageNet100LT
from utils import TwoCropTransform, MemTransform

def normalize(dataset_name):
    normalize_params = {
        'mnist': [(0.1307,), (0.3081,)],
        'cifar10': [(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)],
        'cifar20': [(0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)],
        'cifar100': [(0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)],
        'imagenet': [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)],
        'imagenet50': [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)],
        'imagenet100': [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)],
        'tiny-imagenet-200': [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)],
        'stl10': [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)],
    }
    if dataset_name not in normalize_params.keys():
        mean, std = normalize_params['imagenet']
        print(f'Dataset {dataset_name} does not exist in normalize_params,'
              f' use default normalizations: mean {str(mean)}, std {str(std)}.')
    else:
        mean, std = normalize_params[dataset_name]

    normalize = transforms.Normalize(mean=mean, std=std, inplace=True)
    return normalize

def transform(opt,type):
    normalization = normalize(opt.dataset)
    if type == 'train':
        train_transform = [
            transforms.RandomResizedCrop(size=opt.img_size, scale=(opt.resized_crop_scale, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ]
        if opt.use_gaussian_blur:
            train_transform.append(
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], 0.5)
            )

        train_transform += [transforms.ToTensor(), normalization]
        # train_transform += [transforms.ToTensor()]

        train_transform = transforms.Compose(train_transform)

        train_transform = TwoCropTransform(train_transform)
        return train_transform
    elif type == 'test':
        def resize(image):
            size = (opt.img_size, opt.img_size)
            if image.size == size:
                return image
            return image.resize(size)

        test_transform = []
        if opt.test_resized_crop:
            test_transform += [transforms.Resize(256), transforms.CenterCrop(224)]
        test_transform += [
            resize,
            transforms.ToTensor(),
            normalization
        ]

        test_transform = transforms.Compose(test_transform)
        return test_transform
    elif type == 'mem':
        train_transform = [
            transforms.RandomResizedCrop(size=opt.img_size, scale=(opt.resized_crop_scale, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ]
        if opt.use_gaussian_blur:
            train_transform.append(
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], 0.5)
            )

        train_transform += [transforms.ToTensor(), normalization]
        # train_transform += [transforms.ToTensor()]

        train_transform = transforms.Compose(train_transform)
        def resize(image):
            size = (opt.img_size, opt.img_size)
            if image.size == size:
                return image
            return image.resize(size)

        test_transform = []
        if opt.test_resized_crop:
            test_transform += [transforms.Resize(256), transforms.CenterCrop(224)]
        test_transform += [
            resize,
            transforms.ToTensor(),
            normalization
        ]

        test_transform = transforms.Compose(test_transform)
        return MemTransform(train_transform,test_transform)

def get_dataset(opt,type='train'):
    dataset_name=opt.dataset
    root = opt.data_folder
    tfs = transform(opt,type)
    if dataset_name == 'cifar10':
        dataset = CustomCIFAR10(class_num=opt.cluster_num,max_num=opt.max_num,imb_ratio=opt.imb_ratio,root=root, train=True, transform=tfs, download=False)
    elif dataset_name == 'cifar20':
        dataset = CustomCIFAR100(class_num=opt.cluster_num,max_num=opt.max_num,imb_ratio=opt.imb_ratio,root=root, train=True, transform=tfs, download=True)
    elif dataset_name == 'stl10':
        if type == 'test':
            dataset = CustomSTL10(class_num=opt.cluster_num,max_num=opt.max_num,imb_ratio=opt.imb_ratio,root=root, transform=tfs, download=True)
        else:
            dataset = CustomSTL10(class_num=opt.cluster_num,max_num=opt.max_num,imb_ratio=opt.imb_ratio,root=root, transform=tfs, download=True)

    elif dataset_name == 'tiny-imagenet-200':
        if type == 'test':
            # Tiny ImageNet validation set has 50 images per class
            test_max_num = 50
            dataset = CustomTinyImagenetLT(class_num=opt.cluster_num,
                                           max_num=test_max_num,
                                           imb_ratio=opt.imb_ratio,
                                           root=os.path.join(root, 'val'),
                                           transform=tfs)
        else:
            dataset = CustomTinyImagenetLT(class_num=opt.cluster_num,
                                           max_num=test_max_num,
                                           imb_ratio=opt.imb_ratio,
                                           root=os.path.join(root, 'train'),
                                           transform=tfs)
    elif dataset_name == 'imagenet-lt':
        dataset = CustomImageNetLT(root=root, txt=opt.text_dir, transform=tfs)
    elif dataset_name == 'imagenet100':
        c_num = 100
        if type != 'test':
            dataset = CustomTinyImagenetLT(
                class_num=c_num,
                max_num=opt.max_num,
                imb_ratio=opt.imb_ratio,
                root=os.path.join(root, 'train'),
                transform=tfs
            )
        else:
            dataset = CustomTinyImagenetLT(
                class_num=c_num,
                max_num=50, 
                imb_ratio=opt.imb_ratio,
                root=os.path.join(root, 'val'), 
                transform=tfs
            )
    elif dataset_name == 'imagenet50':
        if type == 'test':
            sub_root = os.path.join(root, 'val')
            dataset = CustomImageNet50LT(
            root=sub_root,
            class_num=opt.cluster_num,
            max_num=50,
            imb_ratio=opt.imb_ratio,
            transform=tfs
            )
        else:
            sub_root = os.path.join(root, 'train')
            dataset = CustomImageNet50LT(
            root=sub_root,
            class_num=opt.cluster_num,
            max_num=opt.max_num,
            imb_ratio=opt.imb_ratio,
            transform=tfs
            )

    return dataset
class logger(object):
    def __init__(self, path, log_name="log.txt", local_rank=0):
        self.path = path
        self.local_rank = local_rank
        self.log_name = log_name

    def info(self, msg):
        if self.local_rank == 0:
            print(msg)
            with open(os.path.join(self.path, self.log_name), 'a') as f:
                f.write(msg + "\n")

def collect_params(*models, exclude_bias_and_bn=True):
    param_list = []
    for model in models:
        for name, param in model.named_parameters():
            param_dict = {
                'name': name,
                'params': param,
            }
            if exclude_bias_and_bn and any(s in name for s in ['bn', 'bias']):
                param_dict.update({'weight_decay': 0., 'lars_exclude': True})
            param_list.append(param_dict)
    return param_list

def cosine_annealing_LR(opt, n_iter):
    step = n_iter - 1
    total_steps = opt.epochs * opt.num_batch
    max_lr = opt.learning_rate
    min_lr = max_lr * opt.learning_eta_min
    warmup_steps = opt.warmup_epochs * opt.num_batch

    if step < warmup_steps:
        # 线性 warmup：从 0 增长到 max_lr
        lr = max_lr * (step + 1) / warmup_steps
    else:
        # 余弦退火：从 max_lr 逐渐降至 min_lr
        anneal_progress = (step - warmup_steps) / (total_steps - warmup_steps)
        lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * anneal_progress))

    return lr

def step_LR(opt, n_iter):
    lr = opt.learning_rate
    epoch = n_iter / opt.num_batch
    if epoch < opt.warmup_epochs:
        # lr = (max_lr - min_lr) * epoch / opt.warmup_epochs + min_lr
        lr = opt.learning_rate * epoch / opt.warmup_epochs
    else:
        for milestone in opt.lr_decay_milestone:
            lr *= opt.lr_decay_gamma if epoch >= milestone else 1.
    return lr

def get_embedding_for_test(model, data_loader, mode='k', loader="test"):
    model.eval()
    local_features = []
    local_labels = []
    for i, (inputs, target, idx) in enumerate(data_loader):
        with torch.no_grad():
            if loader == "test":
                inputs = inputs.to('cuda')
                target = target.to('cuda')
            elif loader == "mem":
                inputs_1, inputs_2, inputs_3 = inputs
                inputs = inputs_1.to('cuda')
                target = target.to('cuda')

            if mode == 'k':
                feature = model.encoder_k(inputs)
                feature = model.projector_k(feature)
            elif mode == 'q':
                feature = model.encoder_q(inputs)
                feature = model.projector_q(feature)
            elif mode == 'p':
                feature = model.encoder_q(inputs)
                feature = model.projector_q(feature)
                feature = model.predictor(feature)
            else:
                raise ValueError(f"Unknown mode: {mode}")
            

            if isinstance(feature, (list, tuple)):
                feature = feature[0]
            if not isinstance(feature, torch.Tensor):
                feature = torch.tensor(feature, device='cuda')
            if feature.dim() > 2:
                feature = torch.flatten(feature, 1)
            local_features.append(feature)
            local_labels.append(target)

    features = torch.cat(local_features, dim=0)
    labels = torch.cat(local_labels, dim=0)
    print(features.shape, labels.shape)
    features = torch.nn.functional.normalize(features, dim=-1)
    return features, labels
