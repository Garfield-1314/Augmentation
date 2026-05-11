import Augmentation_CV
import Augmentation_AL 
import background
import image_mask_AL
import shift_classification

if __name__ == "__main__":
    print("Starting augmentation process...")
    
    # # 示例 1: 生成背景图
    # background.generate_noise_image(size=(160, 160), 
    #                                 save_path='../Datasets/backgrounds/noisy_white_bg.png', 
    #                                 white_background=False, 
    #                                 noise_density=0.05, 
    #                                 show=False)

    # Augmentation_CV.batch_yasuo('../backgrounds', 
    #                         '../Datasets/backgrounds', 
    #                         mode='crop', w=160, h=160)

    # 示例 2: OpenCV 处理
    Augmentation_CV.batch_yasuo('../smartcar26', 
                                '../Datasets/smartcar26_128', 
                                mode='stretch',
                                w=128, h=128)
    
    Augmentation_CV.batch_pixelate('../Datasets/smartcar26_128', 
                                   '../Datasets/smartcar26_128_pixelated', 
                                   size=3)
    
    # 示例 3: Albumentations 增强
    Augmentation_AL.augment_images('../Datasets/smartcar26_128_pixelated', 
                                   '../Datasets/smartcar26_128_pixelated_augmented', 
                                   num_augments=5)

    # 示例 4: 图像合成与增强
    image_mask_AL.batch_overlay(
        backgrounds_dir='../Datasets/backgrounds',
        pics_root='../Datasets/smartcar26_128_pixelated_augmented',
        output_root='../Datasets/smartcar26_128_masked_AL',
        min_scale=0.7,
        max_scale=1.1,
        min_visible=0.9,  # 90%的小图必须位于指定ROI区域内
        num_augments=6
    )
    
    Augmentation_CV.batch_local_illumination('../Datasets/smartcar26_128_masked_AL', 
                                             '../Datasets/smartcar26_128_masked_AL_illumination', 
                                             strength_range=(-40,80), # 增加亮斑数量和强度范围
                                             radius_range=(50, 240), # 设置影响半径范围
                                             num_spots=4) 

    # 示例 5: 数据集划分
    shift_classification.split_dataset(
        source_dir='../Datasets/smartcar26_128_masked_AL_illumination',
        target_dir='../Datasets/smartcar26_dataset',
        train_ratio=0.75,
        val_ratio=0.2,
        test_ratio=0.05,
        seed=749  # 固定随机种子确保可重复性
    )
    print("Augmentation process completed!")
    




