import os
import shutil
import random

def split_dataset(source_dir, target_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=None):
    """
    将源目录中的图片按比例分配到训练集、验证集和测试集
    
    参数：
        source_dir: 包含分类子文件夹的源目录路径
        target_dir: 输出目录路径（会自动创建train/val/test子目录）
        train_ratio: 训练集比例（默认0.7）
        val_ratio: 验证集比例（默认0.2）
        test_ratio: 测试集比例（默认0.1）
        seed: 随机种子（默认None）
    """
    # 验证比例总和为1
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-9, "比例总和必须等于1"
    
    # 设置随机种子
    if seed is not None:
        random.seed(seed)
    
    # 创建目标目录
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(target_dir, split), exist_ok=True)
    
    # 遍历每个类别目录
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        
        if not os.path.isdir(class_path):
            continue
        
        # 获取所有图片文件
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        
        # 打乱文件顺序
        random.shuffle(images)
        total = len(images)
        
        if total == 0:
            print(f"警告: {class_path} 中没有图片文件，跳过处理")
            continue
        
        # 计算分割点
        train_split = int(train_ratio * total)
        val_split = train_split + int(val_ratio * total)
        
        # 分割文件列表
        train_files = images[:train_split]
        val_files = images[train_split:val_split]
        test_files = images[val_split:]
        
        # 复制文件到目标目录
        for split, files in [('train', train_files), 
                           ('val', val_files), 
                           ('test', test_files)]:
            if len(files) == 0:
                continue
            
            dest_dir = os.path.join(target_dir, split, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            
            for f in files:
                src = os.path.join(class_path, f)
                dst = os.path.join(dest_dir, f)
                shutil.copy2(src, dst)
                
        print(f"类别 {class_name} 完成划分: "
             f"{len(train_files)} 训练, "
             f"{len(val_files)} 验证, "
             f"{len(test_files)} 测试")


import Augmentation as Au

def runs():
    # 使用示例
    split_dataset(
        source_dir="dataset/Origin",
        target_dir="dataset/Stage1",
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        seed=34  # 固定随机种子确保可重复性
    )

    root_path = r"dataset\stage1\train"
    save_path = r"dataset\stage2\train"
    Au.YASUO_80(root_path,save_path)
    

    root_path = r"dataset\stage2\train"
    save_path = r"dataset\stage2\train"
    Au.Rotate_90_180_270(root_path,save_path)

    root_path = r"dataset\stage2\train"
    save_path = r"dataset\stage3\train"
    Au.D_dan_B(root_path,save_path)

    root_path = r"dataset\stage1\val"
    save_path = r"dataset\stage2\val"
    Au.YASUO_80(root_path,save_path)
    

    root_path = r"dataset\stage2\val"
    save_path = r"dataset\stage2\val"
    Au.Rotate_90_180_270(root_path,save_path)

    root_path = r"dataset\stage2\val"
    save_path = r"dataset\stage3\val"
    Au.D_dan_B(root_path,save_path)

    root_path = r"dataset\stage1\test"
    save_path = r"dataset\stage1\test"
    Au.YASUO_80(root_path,save_path)
    Au.Rotate_90_180_270(root_path,save_path)
    Au.D_dan_B(root_path,save_path)


if __name__ == "__main__":
    runs()
