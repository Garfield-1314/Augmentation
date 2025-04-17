from xml.dom.minidom import Document
import os
import cv2


def makexml(picPath, txtPath, xmlPath):
    """将YOLO格式的txt标注转换为VOC格式的XML标注，支持JPG和PNG图片"""
    dic = {'0': "object"}  # 可根据需要扩展类别字典

    # 确保输出目录存在
    os.makedirs(xmlPath, exist_ok=True)

    for name in os.listdir(txtPath):
        if not name.endswith('.txt'):
            continue

        # 构建基础文件名和路径
        basename = name[:-4]
        txt_file = os.path.join(txtPath, name)
        
        # 查找对应的图片文件
        img_path = None
        for ext in ['.jpg', '.png']:
            temp_path = os.path.join(picPath, basename + ext)
            if os.path.exists(temp_path):
                img_path = temp_path
                break
        
        if img_path is None:
            print(f"警告：未找到 {basename} 的图片文件，跳过")
            continue

        # 读取图片尺寸
        img = cv2.imread(img_path)
        if img is None:
            print(f"错误：无法读取图片 {img_path}，跳过")
            continue
        Pheight, Pwidth, Pdepth = img.shape

        # 创建XML文档结构
        xmlBuilder = Document()
        annotation = xmlBuilder.createElement("annotation")
        xmlBuilder.appendChild(annotation)

        # 添加文件夹信息
        folder = xmlBuilder.createElement("folder")
        folder.appendChild(xmlBuilder.createTextNode("dataset"))
        annotation.appendChild(folder)

        # 添加文件名（带正确扩展名）
        filename = xmlBuilder.createElement("filename")
        filename.appendChild(xmlBuilder.createTextNode(os.path.basename(img_path)))
        annotation.appendChild(filename)

        # 添加图片尺寸信息
        size = xmlBuilder.createElement("size")
        for tag, value in [('width', Pwidth), ('height', Pheight), ('depth', Pdepth)]:
            element = xmlBuilder.createElement(tag)
            element.appendChild(xmlBuilder.createTextNode(str(value)))
            size.appendChild(element)
        annotation.appendChild(size)

        # 处理标注信息
        with open(txt_file, 'r') as txtFile:
            for line in txtFile:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue

                # 创建object节点
                obj = xmlBuilder.createElement("object")
                for tag, value in [('name', dic.get(parts[0], 'unknown')),
                                 ('pose', 'Unspecified'),
                                 ('truncated', '0'),
                                 ('difficult', '0')]:
                    element = xmlBuilder.createElement(tag)
                    element.appendChild(xmlBuilder.createTextNode(value))
                    obj.appendChild(element)

                # 转换YOLO坐标到VOC坐标
                x_center = float(parts[1]) * Pwidth
                y_center = float(parts[2]) * Pheight
                w = float(parts[3]) * Pwidth
                h = float(parts[4]) * Pheight

                bndbox = xmlBuilder.createElement("bndbox")
                box_data = {
                    'xmin': int(x_center - w/2),
                    'ymin': int(y_center - h/2),
                    'xmax': int(x_center + w/2),
                    'ymax': int(y_center + h/2)
                }
                for tag in ['xmin', 'ymin', 'xmax', 'ymax']:
                    element = xmlBuilder.createElement(tag)
                    element.appendChild(xmlBuilder.createTextNode(str(box_data[tag])))
                    bndbox.appendChild(element)
                obj.appendChild(bndbox)
                annotation.appendChild(obj)

        # 保存XML文件
        output_path = os.path.join(xmlPath, f"{basename}.xml")
        with open(output_path, 'w', encoding='utf-8') as f:
            xmlBuilder.writexml(f, indent='\t', addindent='\t', newl='\n', encoding='utf-8')


if __name__ == "__main__":
    # 使用示例（路径需要根据实际情况修改）
    makexml(
        picPath="VOCdevkit/VOC2007/JPEGImages/",
        txtPath="VOCdevkit/VOC2007/YOLOLabels/",
        xmlPath="VOCdevkit/VOC2007/Annotations/"
    )