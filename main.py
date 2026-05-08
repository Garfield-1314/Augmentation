import Augmentation_CV
import Augmentation_AL 
import background
import image_mask_AL
import shift_classification

if __name__ == "__main__":
    print("Starting augmentation process...")
    
    # 示例 1: 生成背景图
    print("Generating background images...")
    background.generate_noise_image(size=(160, 240), 
                                    save_path='../Datasets/backgrounds/noisy_white_bg.png', 
                                    white_background=False, 
                                    noise_density=0.05, 
                                    show=False)

    # 示例 2: OpenCV 处理
    print("Compressing images to 160x160...")
    Augmentation_CV.batch_yasuo('../smartcar26', 
                                '../Datasets/smartcar26_160', 
                                160, 160)
    
    Augmentation_CV.batch_pixelate('../Datasets/smartcar26_160', 
                                   '../Datasets/smartcar26_160_pixelated', 
                                   size=3)
    
    # 示例 3: Albumentations 增强
    print("Applying advanced augmentations...")
    Augmentation_AL.augment_images('../Datasets/smartcar26_160_pixelated', 
                                   '../Datasets/smartcar26_160_pixelated_augmented', 
                                   num_augments=5)

    # 示例 4: 图像合成与增强
    print("Starting image composition and augmentation...")
    image_mask_AL.batch_overlay(
        backgrounds_dir='../Datasets/backgrounds',
        pics_root='../Datasets/smartcar26_160_pixelated_augmented',
        output_root='../Datasets/smartcar26_160_masked_AL',
        min_scale=0.7,
        max_scale=1.1,
        min_visible=0.9,  # 90%的小图必须位于指定ROI区域内
        num_augments=5
    )
    
    print("Adding local illumination augmentation...")
    Augmentation_CV.batch_local_illumination('../Datasets/smartcar26_160_masked_AL', 
                                             '../Datasets/smartcar26_160_masked_AL_illumination', 
                                             strength_range=(-50,100), # 添加局部光照增强，增加亮斑数量和强度范围
                                             radius_range=(40, 160), # 添加局部光照增强，设置影响半径范围
                                             num_spots=4) # 添加局部光照增强  

    # 示例 5: 数据集划分
    print("Splitting dataset into train/val/test...")
    shift_classification.split_dataset(
        source_dir='../Datasets/smartcar26_160_masked_AL_illumination',
        target_dir='../Datasets/smartcar26_dataset',
        train_ratio=0.75,
        val_ratio=0.2,
        test_ratio=0.05,
        seed=749  # 固定随机种子确保可重复性
    )
    print("Augmentation process completed!")
    




