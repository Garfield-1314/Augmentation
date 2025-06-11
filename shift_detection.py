import os
import random
import shutil
from sklearn.model_selection import train_test_split

def split_yolo_dataset(dataset_root, 
                      train_ratio=0.8, 
                      copy_files=True, 
                      random_seed=42):
    """
    划分YOLO数据集为训练集和验证集
    
    参数:
    dataset_root: 数据集根目录路径 (包含images和labels文件夹)
    train_ratio: 训练集比例 (默认0.8)
    copy_files: 是否复制文件到新目录 (True) 还是仅创建索引文件 (False)
    random_seed: 随机种子 (确保可重复结果)
    """
    
    # 定义路径
    img_dir = os.path.join(dataset_root, 'images')
    label_dir = os.path.join(dataset_root, 'labels')
    
    # 验证目录存在
    if not os.path.exists(img_dir):
        raise ValueError(f"图片目录不存在: {img_dir}")
    if not os.path.exists(label_dir):
        raise ValueError(f"标签目录不存在: {label_dir}")
    
    # 获取所有图片文件名（不带扩展名）
    all_images = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    base_names = [os.path.splitext(f)[0] for f in all_images]
    
    # 检查对应的标签文件
    valid_base_names = []
    for base in base_names:
        label_path = os.path.join(label_dir, base + '.txt')
        if os.path.exists(label_path):
            valid_base_names.append(base)
        else:
            print(f"警告: 缺少标签文件 {base}.txt")
    
    # 数据集划分
    train_names, val_names = train_test_split(
        valid_base_names, 
        train_size=train_ratio, 
        random_state=random_seed
    )
    
    print(f"总样本数: {len(valid_base_names)}")
    print(f"训练集数量: {len(train_names)}")
    print(f"验证集数量: {len(val_names)}")
    
    # 创建目录结构
    if copy_files:
        for folder in ['train', 'val']:
            os.makedirs(os.path.join(dataset_root, folder, 'images'), exist_ok=True)
            os.makedirs(os.path.join(dataset_root, folder, 'labels'), exist_ok=True)
    
    # 准备索引文件
    train_txt = os.path.join(dataset_root, 'train.txt')
    val_txt = os.path.join(dataset_root, 'val.txt')
    
    with open(train_txt, 'w') as f_train, open(val_txt, 'w') as f_val:
        # 处理训练集
        for name in train_names:
            src_img = os.path.join(img_dir, name + get_extension(img_dir, name))
            src_label = os.path.join(label_dir, name + '.txt')
            
            if copy_files:
                dest_img = os.path.join(dataset_root, 'train', 'images', os.path.basename(src_img))
                dest_label = os.path.join(dataset_root, 'train', 'labels', name + '.txt')
                shutil.copy2(src_img, dest_img)
                shutil.copy2(src_label, dest_label)
                f_train.write(dest_img + '\n')
            else:
                f_train.write(src_img + '\n')
        
        # 处理验证集
        for name in val_names:
            src_img = os.path.join(img_dir, name + get_extension(img_dir, name))
            src_label = os.path.join(label_dir, name + '.txt')
            
            if copy_files:
                dest_img = os.path.join(dataset_root, 'val', 'images', os.path.basename(src_img))
                dest_label = os.path.join(dataset_root, 'val', 'labels', name + '.txt')
                shutil.copy2(src_img, dest_img)
                shutil.copy2(src_label, dest_label)
                f_val.write(dest_img + '\n')
            else:
                f_val.write(src_img + '\n')
    
    print("数据集划分完成!")
    print(f"训练集索引: {train_txt}")
    print(f"验证集索引: {val_txt}")
    if copy_files:
        print(f"文件已复制到: {os.path.join(dataset_root, 'train')} 和 {os.path.join(dataset_root, 'val')}")

def get_extension(img_dir, base_name):
    """查找图片文件的实际扩展名"""
    for ext in ['.png', '.jpg', '.jpeg']:
        if os.path.exists(os.path.join(img_dir, base_name + ext)):
            return ext
    raise FileNotFoundError(f"找不到图片文件: {base_name}")

if __name__ == "__main__":
    # 设置数据集根目录路径
    DATASET_ROOT = './text4'  # 修改为你的数据集路径
    
    # 执行划分
    split_yolo_dataset(
        dataset_root=DATASET_ROOT,
        train_ratio=0.8,        # 80%训练集，20%验证集
        copy_files=True,         # 复制文件到新目录
        random_seed=42           # 随机种子
    )