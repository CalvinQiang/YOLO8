from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 加载预训练模型
    model = YOLO('../0-models/yolov8n.pt')

    # 2. 对单张图片进行预测
    results = model('./cat.jpg')  # 请将路径替换为你自己的图片路径

    # 3. 获取第一张图片的结果
    result = results[0]

    print("=== Results对象基本信息 ===")
    print(f"检测到 {len(result.boxes)} 个对象")
    print(f"原始图像路径: {result.path}")
    print(f"原始图像尺寸: {result.orig_shape}")
    print(f"处理速度: {result.speed}")

    print("\n=== 详细检测信息 ===")
    # 将张量转移到CPU并转换为numpy数组（如果模型在GPU上运行需要这一步）[5](@ref)
    boxes_xyxy = result.boxes.xyxy.cpu().numpy()  # 坐标
    conf_scores = result.boxes.conf.cpu().numpy()  # 置信度
    class_indices = result.boxes.cls.cpu().numpy().astype(int)  # 类别索引

    # 遍历每个检测到的对象
    for i, (box, conf, cls_idx) in enumerate(zip(boxes_xyxy, conf_scores, class_indices)):
        x1, y1, x2, y2 = box
        cls_name = model.names[cls_idx]  # 获取类别名称[5](@ref)

        print(f"对象 {i + 1}:")
        print(f"  类别: {cls_name} (ID: {cls_idx})")
        print(f"  置信度: {conf:.4f}")
        print(f"  坐标: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
        print(f"  中心点: [{(x1 + x2) / 2:.1f}, {(y1 + y2) / 2:.1f}], 宽度: {x2 - x1:.1f}, 高度: {y2 - y1:.1f}")
        print("-" * 40)

    # 4. 展示和保存结果
    result.show()  # 显示带检测框的图片
    result.save('./output_image.jpg')  # 保存结果图片
