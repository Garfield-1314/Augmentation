import os
import cv2
import numpy as np
from tqdm import tqdm
from albumentations import (
    Compose, ElasticTransform, OpticalDistortion, Rotate,
    RGBShift, RandomBrightnessContrast, HueSaturationValue, MotionBlur, BboxParams
)
from modules.Augmentation_CV import LocalIllumination

def get_transforms(
    p_elastic=0.25, elastic_alpha=1.2, elastic_sigma=25,
    p_optical=0.25, optical_distort=0.25,
    p_rotate=0.25, rotate_limit=2,
    p_rgb=0.25, r_shift=10, g_shift=10, b_shift=10,
    p_brightness=0.5, brightness_limit=0.25, contrast_limit=0.15,
    p_hue=0.25, hue_shift=10, sat_shift=20, val_shift=10,
    p_motion=0.25, motion_blur_limit=3
):
    """根据详细参数生成增强管道"""
    # 局部曝光/光照在这里不通过 albumentations，因为它是一个外部 OpenCV 函数
    return Compose(
        [
            # 弹性变形
            ElasticTransform(p=p_elastic, alpha=elastic_alpha, sigma=elastic_sigma),
            # 光学畸变
            OpticalDistortion(p=p_optical, distort_limit=optical_distort, interpolation=cv2.INTER_NEAREST),
            # 随机旋转
            Rotate(limit=rotate_limit, p=p_rotate, border_mode=cv2.INTER_NEAREST),
            # RGB通道偏移
            RGBShift(r_shift_limit=r_shift, g_shift_limit=g_shift, b_shift_limit=b_shift, p=p_rgb),
            # 亮度对比度调整
            RandomBrightnessContrast(p=p_brightness, 
                                     brightness_limit=(-brightness_limit, brightness_limit), 
                                     contrast_limit=(-contrast_limit, contrast_limit)),
            # 色相饱和度调整
            HueSaturationValue(hue_shift_limit=hue_shift, sat_shift_limit=sat_shift, val_shift_limit=val_shift, p=p_hue),
            # 动态模糊
            MotionBlur(p=p_motion, blur_limit=(3, int(motion_blur_limit)))
        ],
        bbox_params=BboxParams(coord_format='yolo', min_visibility=0.4, min_area=8)
    )

def batch_yolo_augment(input_images_dir, output_images_dir, input_labels_dir=None, output_labels_dir=None, Au_num=5, 
                       p_illumination=0.3, illumination_min=-100, illumination_max=150, illumination_spots=3, **kwargs):
    """
    专门为 UI 提供的批量 YOLO 增强接口
    """
    # 路径兼容处理：如果传入的是根目录，尝试进入 images 子目录
    print(f"DEBUG: Processing input_images_dir: {input_images_dir}")
    if os.path.exists(os.path.join(input_images_dir, "images")):
        actual_input_images = os.path.join(input_images_dir, "images")
        if not input_labels_dir:
            input_labels_dir = os.path.join(input_images_dir, "labels")
        print(f"DEBUG: Found images folder, using: {actual_input_images}")
    else:
        # 检查当前目录下是否有图片
        img_files_check = [f for f in os.listdir(input_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if img_files_check:
            actual_input_images = input_images_dir
            if not input_labels_dir:
                # 尝试猜测 labels 路径（如果当前文件夹叫 images，尝试替换为 labels）
                if "images" in input_images_dir.lower():
                    input_labels_dir = input_images_dir.lower().replace('images', 'labels')
                else:
                    # 否则就在同级找 labels 文件夹
                    parent = os.path.dirname(input_images_dir)
                    input_labels_dir = os.path.join(parent, "labels")
            print(f"DEBUG: Directly using folder (found {len(img_files_check)} images): {actual_input_images}")
        else:
            print(f"DEBUG: No images found in {input_images_dir}")
            return # 直接返回，不抛异常防止 UI 崩溃，但控制台能看到原因

    # 输出路径处理
    actual_output_images = os.path.join(output_images_dir, "images")
    actual_output_labels = os.path.join(output_images_dir, "labels")
    
    # 自动创建输出子文件夹
    os.makedirs(actual_output_images, exist_ok=True)
    os.makedirs(actual_output_labels, exist_ok=True)
    
    print(f"DEBUG: Output images folder created/verified: {actual_output_images}")
    print(f"DEBUG: Output labels folder created/verified: {actual_output_labels}")
    print(f"DEBUG: Input labels path: {input_labels_dir}")

    transform = get_transforms(**kwargs)

    img_files = [f for f in os.listdir(actual_input_images) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_file in tqdm(img_files, desc="YOLO Augmenting"):
        img_path = os.path.join(actual_input_images, img_file)
        base_name = os.path.splitext(img_file)[0]
        # 统一路径分隔符，防止 Windows 下出现混合斜杠导致判断错误
        txt_path = os.path.join(input_labels_dir, base_name + '.txt').replace('\\', '/')
        img_path = img_path.replace('\\', '/')

        image = cv2.imread(img_path)
        if image is None: 
            print(f"WARNING: Could not read image {img_path}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bboxes = []
        # 调试信息：打印正在寻找的标签路径
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        bboxes.append([float(x) for x in parts[1:]] + [int(parts[0])])
        else:
            print(f"DEBUG: Label not found for {img_file}, path tried: {txt_path}")

        # 生成并保存
        for i in range(Au_num + 1):
            suffix = f"_aug{i}" if i > 0 else ""
            try:
                # 即使没有 bboxes (bboxes=[])，transform 也会运行并生成图
                res = transform(image=image, bboxes=bboxes)
                
                # --- 集成 CV 的局部曝光功能 ---
                augmented_image = res['image']
                if np.random.random() < p_illumination:
                    # 注意：LocalIllumination 期望的是 BGR 格式，所以我们需要转换
                    # 但其实它内部是直接处理像素值，只要输入输出一致即可。
                    # 为了安全，我们传递 RGB 并由于其内部是对称处理，结果也是 RGB
                    augmented_image = LocalIllumination(
                        augmented_image, 
                        num_spots=illumination_spots, 
                        strength_range=(illumination_min, illumination_max)
                    )
                
                # 保存图
                out_img_name = f"{base_name}{suffix}.jpg"
                save_img_path = os.path.join(actual_output_images, out_img_name)
                cv2.imwrite(save_img_path, cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))
                
                # 保存标签
                out_txt_name = f"{base_name}{suffix}.txt"
                save_txt_path = os.path.join(actual_output_labels, out_txt_name)
                with open(save_txt_path, 'w', encoding='utf-8') as f:
                    for b in res['bboxes']:
                        f.write(f"{int(b[4])} {' '.join([f'{x:.6f}' for x in b[:4]])}\n")
                
            except Exception as e:
                print(f"ERROR augmenting {img_file} copy {i}: {e}")

# 原有逻辑保留用于直接运行脚本
def process_dataset(augment=True, Au_num=10):
    # ... 保持不变或根据需要更新
    pass


# 路径配置
base_dir = {
    'images': '../yolo/images',
    'labels': '../yolo/labels'
}

output_dir = {
    'images': '../yolo2',
    'labels': '../yolo2'
}

def process_dataset(augment=True, Au_num=10):
    """处理整个数据集"""
    # 创建输出目录
    os.makedirs(output_dir['images'], exist_ok=True)
    os.makedirs(output_dir['labels'], exist_ok=True)
    
    # 选择变换器
    transform = get_transforms() if augment else Compose([], bbox_params=BboxParams(coord_format='yolo'))
    
    # 遍历原始图像
    img_folder = base_dir['images']
    total_files = len([f for f in os.listdir(img_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    with tqdm(total=total_files, desc='Processing images', unit='img') as pbar:
        for img_file in os.listdir(img_folder):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            # 构造路径
            img_path = os.path.join(img_folder, img_file)
            base_name = os.path.splitext(img_file)[0]
            txt_path = os.path.join(base_dir['labels'], base_name + '.txt')

            # 读取数据
            image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            
            # 读取标签
            bboxes = []
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id, xc, yc, w, h = map(float, parts)
                            bboxes.append([xc, yc, w, h, int(class_id)])
            
            # 应用增强
            try:
                augmented = transform(image=image, bboxes=bboxes)
            except Exception as e:
                print(f"\nError processing {img_file}: {str(e)}")
                continue

            # 保存增强结果（原始图/基准图）
            save_augmented(
                augmented['image'],
                augmented['bboxes'],
                img_file,
                copy_number=0
            )

            # 随机生成增强副本
            if augment:
                for copy_idx in range(Au_num):
                    try:
                        augmented_copy = transform(image=image, bboxes=bboxes)
                        save_augmented(
                            augmented_copy['image'],
                            augmented_copy['bboxes'],
                            img_file,
                            copy_number=copy_idx + 1
                        )
                    except Exception as e:
                        print(f"\nError generating augmented copy {copy_idx+1} for {img_file}: {str(e)}")

            pbar.update(1)

def save_augmented(image, bboxes, orig_filename, copy_number=0):
    """保存增强后的数据"""
    # 生成唯一文件名
    base_name = os.path.splitext(orig_filename)[0]
    suffix = f"_aug{copy_number}" if copy_number > 0 else ""
    new_filename = f"{base_name}{suffix}.jpg"
    
    # 保存图像
    cv2.imwrite(
        os.path.join(output_dir['images'], new_filename),
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    )
    
    # 保存标签
    label_path = os.path.join(output_dir['labels'], f"{base_name}{suffix}.txt")
    with open(label_path, 'w') as f:
        for bbox in bboxes:
            class_id = int(bbox[4])
            coords = [f"{x:.6f}" for x in bbox[:4]]
            f.write(f"{class_id} {' '.join(coords)}\n")

# 执行处理
if __name__ == "__main__":
    # 直接对整个文件夹进行增强（生成10个增强版本）
    process_dataset(augment=True, Au_num=10)
    
    print("All data processed!")