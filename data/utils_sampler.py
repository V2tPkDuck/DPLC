import torch
import numpy as np

def make_imb_data(max_num, class_num, gamma, flag=1, flag_LT=0):
    mu = np.power(1 / gamma, 1 / (class_num - 1))
    class_num_list = []
    for i in range(class_num):
        if i == (class_num - 1):
            class_num_list.append(int(max_num / gamma))
        else:
            class_num_list.append(int(max_num * np.power(mu, i)))
    if flag == 0 and flag_LT == 1:
        class_num_list = list(reversed(class_num_list))

    return class_num_list

def train_split(labels, n_per_class):
    class_indices = {}
    for idx, target in enumerate(labels):
        if target not in class_indices:
            class_indices[target] = []
        class_indices[target].append(idx)
    return _sample_indices_from_class_indices(class_indices, n_per_class)

def _sample_indices_from_class_indices(class_indices, class_num_list):
    sampled_indices = []
    for class_idx, target_num in enumerate(class_num_list):
        if class_idx not in class_indices:
            continue

        current_class_indices = class_indices[class_idx]
        sample_size = min(target_num, len(current_class_indices))
        chosen_indices = np.random.choice(
            current_class_indices,
            sample_size,
            replace=False,
        )
        sampled_indices.extend(chosen_indices.tolist())

    return sampled_indices

def superclass(targets):
    targets = np.array(targets)
    super_classes = [
        [72, 4, 95, 30, 55],
        [73, 32, 67, 91, 1],
        [92, 70, 82, 54, 62],
        [16, 61, 9, 10, 28],
        [51, 0, 53, 57, 83],
        [40, 39, 22, 87, 86],
        [20, 25, 94, 84, 5],
        [14, 24, 6, 7, 18],
        [43, 97, 42, 3, 88],
        [37, 17, 76, 12, 68],
        [49, 33, 71, 23, 60],
        [15, 21, 19, 31, 38],
        [75, 63, 66, 64, 34],
        [77, 26, 45, 99, 79],
        [11, 2, 35, 46, 98],
        [29, 93, 27, 78, 44],
        [65, 50, 74, 36, 80],
        [56, 52, 47, 59, 96],
        [8, 58, 90, 13, 48],
        [81, 69, 41, 89, 85],
    ]
    import copy
    copy_targets = copy.deepcopy(targets)
    for i in range(len(super_classes)):
        for j in super_classes[i]:
            targets[copy_targets == j] = i
    return targets