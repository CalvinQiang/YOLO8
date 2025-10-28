from ultralytics import YOLO

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # 1. 加载一个预训练模型（这里使用轻量级的 yolov8n）
    model = YOLO('yolov8n.pt')

    # 2. 对单张图片进行预测
    results = model('./cat.jpg')  # 请将路径替换为你自己的图片路径

    # 3. 展示结果（会直接显示带检测框的图片）
    results[0].show()

    # 4. 也可以保存结果图片
    results[0].save('./output_image.jpg')

