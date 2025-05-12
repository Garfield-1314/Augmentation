import albumentations as A
import cv2
import os
from tqdm import tqdm

# 定义增强变换管道
train_transform = A.Compose(
    [
   # 弹性变形（添加边界反射模式）
    A.ElasticTransform(
        p=0.25,
        alpha=1.2,
        sigma=25,
    ),
    # 光学畸变（添加黑色填充）
    A.OpticalDistortion(
        p=0.25,
        distort_limit=0.25,
        interpolation=cv2.INTER_NEAREST,
        mask_interpolation=cv2.INTER_NEAREST
        # fill_value=(255, 255, 255)
    ),
    # 随机旋转（镜像边界处理）
    A.Rotate(
        limit=15,
        p=0.6,
        border_mode=cv2.INTER_NEAREST
    ),
    # RGB通道偏移
    A.RGBShift(
        r_shift_limit=15,
        g_shift_limit=15,
        b_shift_limit=15,
        p=0.3
    ),
    # 亮度对比度调整
    A.RandomBrightnessContrast(
        p=0.8,
        brightness_limit=(-0.3, 0.3),
        contrast_limit=(-0.15, 0.15)
    ),
    # 色相饱和度调整
    A.HueSaturationValue(
        hue_shift_limit=15,
        sat_shift_limit=25,
        val_shift_limit=15,
        p=0.4
    ),
    # 动态模糊（使用随机核大小）
    A.MotionBlur(
        p=0.3,
        blur_limit=(3, 9)
    )
    ],
    bbox_params=A.BboxParams(
        format='yolo',
        min_visibility=0.4,  # 过滤可见性低于40%的bbox
        min_area=8,         # 过滤面积小于8像素的bbox
    )
)

# 验证集不进行增强（仅示例保留结构）
val_transform = A.Compose(
    [
   # 弹性变形（添加边界反射模式）
    A.ElasticTransform(
        p=0.25,
        alpha=1.2,
        sigma=25,
    ),
    # 光学畸变（添加黑色填充）
    A.OpticalDistortion(
        p=0.25,
        distort_limit=0.25,
    ),
    # 随机旋转（镜像边界处理）
    A.Rotate(
        limit=15,
        p=0.6,
        border_mode=cv2.BORDER_REFLECT_101
    ),
    # RGB通道偏移
    A.RGBShift(
        r_shift_limit=15,
        g_shift_limit=15,
        b_shift_limit=15,
        p=0.3
    ),
    # 亮度对比度调整
    A.RandomBrightnessContrast(
        p=0.8,
        brightness_limit=(-0.3, 0.3),
        contrast_limit=(-0.15, 0.15)
    ),
    # 色相饱和度调整
    A.HueSaturationValue(
        hue_shift_limit=15,
        sat_shift_limit=25,
        val_shift_limit=15,
        p=0.4
    ),
    # 动态模糊（使用随机核大小）
    A.MotionBlur(
        p=0.3,
        blur_limit=(3, 7)
    )
    ],
    bbox_params=A.BboxParams(format='yolo')
)

# 路径配置
base_dir = {
    'images': {
        'train': '../biyesheji/dataset/images/train',
        'val': '../biyesheji/dataset/images/val'
    },
    'labels': {
        'train': '../biyesheji/dataset/labels/train',
        'val': '../biyesheji/dataset/labels/val'
    }
}

output_dir = {
    'images': {
        'train': 'augmented/images/train',
        'val': 'augmented/images/val'
    },
    'labels': {
        'train': 'augmented/labels/train',
        'val': 'augmented/labels/val'
    }
}

def process_split(split_name, augment=True,Au_num = 10):
    """处理单个数据集分割"""
    # 创建输出目录
    os.makedirs(output_dir['images'][split_name], exist_ok=True)
    os.makedirs(output_dir['labels'][split_name], exist_ok=True)
    
    # 选择变换器
    transform = train_transform if augment else val_transform
    
    # 遍历原始图像
    img_folder = base_dir['images'][split_name]
    total_files = len(os.listdir(img_folder))
    
    with tqdm(total=total_files, desc=f'Processing {split_name}', unit='img') as pbar:
        for img_file in os.listdir(img_folder):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            # 构造路径
            img_path = os.path.join(img_folder, img_file)
            base_name = os.path.splitext(img_file)[0]
            txt_path = os.path.join(base_dir['labels'][split_name], base_name + '.txt')

            # 读取数据
            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
            
            # 读取标签
            bboxes = []
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    for line in f:
                        class_id, xc, yc, w, h = map(float, line.strip().split())
                        bboxes.append([xc, yc, w, h, int(class_id)])
            
            # 应用增强
            try:
                augmented = transform(image=image, bboxes=bboxes)
            except Exception as e:
                print(f"\nError processing {img_file}: {str(e)}")
                continue

            # 保存增强结果
            if augment or len(augmented['bboxes']) > 0:  # 验证集保留所有样本
                save_augmented(
                    augmented['image'],
                    augmented['bboxes'],
                    img_file,
                    split_name,
                    copy_number=0
                )

                # 随机生成增强副本
                if augment:
                    for copy_idx in range(Au_num):  # 每个样本生成2个增强副本
                        augmented_copy = train_transform(image=image, bboxes=bboxes)
                        save_augmented(
                            augmented_copy['image'],
                            augmented_copy['bboxes'],
                            img_file,
                            split_name,
                            copy_number=copy_idx+1
                        )

            pbar.update(1)

def save_augmented(image, bboxes, orig_filename, split_name, copy_number=0):
    """保存增强后的数据"""
    # 生成唯一文件名
    base_name = os.path.splitext(orig_filename)[0]
    suffix = f"_aug{copy_number}" if copy_number > 0 else ""
    new_filename = f"{base_name}{suffix}.jpg"
    
    # 保存图像
    cv2.imwrite(
        os.path.join(output_dir['images'][split_name], new_filename),
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    )
    
    # 保存标签
    with open(os.path.join(output_dir['labels'][split_name], f"{base_name}{suffix}.txt"), 'w') as f:
        for bbox in bboxes:
            class_id = int(bbox[4])
            coords = [f"{x:.6f}" for x in bbox[:4]]
            f.write(f"{class_id} {' '.join(coords)}\n")

# 执行处理
if __name__ == "__main__":
    # 训练集增强（生成10个增强版本）
    process_split('train', augment=True , Au_num = 10)
    
    # 验证集原样复制（可选）
    process_split('val', augment=True , Au_num = 10)
    
    print("All data processed!")