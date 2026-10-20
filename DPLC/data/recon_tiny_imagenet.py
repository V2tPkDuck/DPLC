import os
import shutil

# UPDATE THIS PATH to your actual tiny-imagenet-200/val directory
val_dir = './Datasets/tiny-imagenet-200/tiny-imagenet-200/val'
val_annotations = os.path.join(val_dir, 'val_annotations.txt')
val_images_dir = os.path.join(val_dir, 'images')

with open(val_annotations, 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        img_name, class_id = parts[0], parts[1]
        
        class_dir = os.path.join(val_dir, class_id)
        os.makedirs(class_dir, exist_ok=True)
        
        src = os.path.join(val_images_dir, img_name)
        dst = os.path.join(class_dir, img_name)
        if os.path.exists(src):
            shutil.move(src, dst)

print("Validation directory properly formatted!")