import os

# 设置图片和标签文件夹路径
images_dir = 'images'  # 图片文件夹路径
labels_dir = 'labels'  # XML标签文件夹路径

# 支持的图片文件扩展名  
allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

def clean_orphan_images(img_dir, label_dir):
    """删除没有对应XML文件的图片"""
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        
        # 跳过目录和非文件项
        if not os.path.isfile(img_path):
            continue
        
        # 解析文件名和扩展名
        filename, ext = os.path.splitext(img_name)
        ext = ext.lower()  # 统一小写处理
        
        # 跳过非图片文件
        if ext not in allowed_extensions:
            print(f"跳过非图片文件: {img_name}")
            continue
        
        # 构建对应的XML路径
        xml_path = os.path.join(label_dir, f"{filename}.xml")
        
        # 删除没有对应XML的图片
        if not os.path.exists(xml_path):
            os.remove(img_path)
            print(f"已删除无对应XML的图片: {img_path}")

if __name__ == "__main__":
    images_dir = "./smartcar/JPEGImages"
    labels_dir = "./smartcar/Annotations"
    clean_orphan_images(images_dir, labels_dir)
    print("清理完成！")