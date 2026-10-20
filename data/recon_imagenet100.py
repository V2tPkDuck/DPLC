import os
import shutil
import random
from tqdm import tqdm

base_dir = "./Datasets/imagenet100/"
n_train = 1300
n_val = 500

train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

class_dirs = [d for d in os.listdir(base_dir)
              if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('n')]

print(f"发现 {len(class_dirs)} 个类别文件夹")

random.seed(42)

for cls in tqdm(class_dirs, desc="处理类别"):
    cls_path = os.path.join(base_dir, cls)
    images = sorted([f for f in os.listdir(cls_path) 
                     if os.path.isfile(os.path.join(cls_path, f))])
    
    if len(images) < n_train + n_val:
        print(f"警告：类别 {cls} 只有 {len(images)} 张图片，不足 {n_train + n_val}，将全部使用并均分")
        n_available = len(images)
        train_imgs = images[:min(n_available, n_train)]
        val_imgs = images[min(n_available, n_train):] if n_available > n_train else []
    else:
        random.shuffle(images)
        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
    
    target_train_cls = os.path.join(train_dir, cls)
    target_val_cls = os.path.join(val_dir, cls)
    os.makedirs(target_train_cls, exist_ok=True)
    os.makedirs(target_val_cls, exist_ok=True)
    
    for img in train_imgs:
        src = os.path.join(cls_path, img)
        dst = os.path.join(target_train_cls, img)
        shutil.move(src, dst) 

    for img in val_imgs:
        src = os.path.join(cls_path, img)
        dst = os.path.join(target_val_cls, img)
        shutil.move(src, dst)

print("重组完成！")
for cls in class_dirs:
    cls_path = os.path.join(base_dir, cls)
    if os.path.isdir(cls_path) and len(os.listdir(cls_path)) == 0:
        os.rmdir(cls_path)

print("原始空文件夹已清理。")