import os
import shutil
import random
from tqdm import tqdm
from glob import glob

def split_dataset(
    source_dir,
    train_dir,
    test_dir,
    train_ratio=0.8,
    test_ratio=0.2,
    seed=None
):
    """
    修复文件重复问题的版本
    """
    assert abs((train_ratio + test_ratio) - 1.0) < 1e-9, "比例总和必须等于1"
    
    if seed is not None:
        random.seed(seed)
    
    # 改进的文件收集方式（去重）
    def get_unique_files(path, exts):
        seen = set()
        files = []
        for ext in exts:
            for pattern in [f'*.{ext}', f'*.{ext.upper()}']:
                for f in glob(os.path.join(path, '**', pattern), recursive=True):
                    # 标准化路径用于去重（特别处理Windows大小写问题）
                    norm_path = os.path.normcase(f)
                    if norm_path not in seen:
                        seen.add(norm_path)
                        files.append(f)
        return files

    # 遍历每个类别目录
    classes = [d for d in os.listdir(source_dir) 
              if os.path.isdir(os.path.join(source_dir, d))]
    
    for class_name in tqdm(classes, desc="Processing classes"):
        class_path = os.path.join(source_dir, class_name)
        
        # 获取去重后的文件列表
        images = get_unique_files(class_path, ['png', 'jpg', 'jpeg', 'gif', 'bmp'])
        
        if not images:
            print(f"警告: {class_path} 中没有图片文件，跳过处理")
            continue
        
        # 转换为相对路径
        rel_images = [os.path.relpath(p, class_path) for p in images]
        
        random.shuffle(rel_images)
        total = len(rel_images)
        
        train_split = round(train_ratio * total)
        test_split = total - train_split
        
        train_files = rel_images[:train_split]
        test_files = rel_images[train_split:]
        
        # 复制文件函数封装
        def copy_files(files, dest_root):
            for rel_path in tqdm(files, desc=f"Copying to {os.path.basename(dest_root)}"):
                src = os.path.join(class_path, rel_path)
                dst = os.path.join(dest_root, class_name, rel_path)
                
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                
                # 仅在文件不存在时复制（避免重复）
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                else:
                    print(f"警告: 跳过重复文件 {dst}")
        
        # 执行复制
        copy_files(train_files, train_dir)
        copy_files(test_files, test_dir)
        
        print(f"\n类别 {class_name} 划分完成: "
             f"{len(train_files)} 训练, {len(test_files)} 测试")


import Augmentation as Au

def runs():
    split_dataset(
        source_dir="dataset/Origin",
        train_dir="dataset/stage1",
        test_dir="dataset/test",
        train_ratio=0.9,
        test_ratio=0.1,
        seed=66
    )

    root_path = r"dataset\stage1"
    save_path = r"dataset\stage2"
    Au.YASUO_80(root_path,save_path)

    root_path = r"dataset\stage2"
    save_path = r"dataset\stage2"
    Au.Rotate_90_180_270(root_path,save_path)


    root_path = r"dataset\stage2"
    save_path = r"dataset\stage3"
    Au.D_dan_B(root_path,save_path)

    root_path = r"dataset\test"
    save_path = r"dataset\test"
    Au.YASUO_80(root_path,save_path)
    Au.Rotate_90_180_270(root_path,save_path)
    Au.D_dan_B(root_path,save_path)


if __name__ == "__main__":
    runs()
