from ultralytics import YOLO

if __name__ == '__main__':
    # 加载模型
    model = YOLO('../0-models/yolov8n.pt')

    # 指定输入视频路径和输出路径
    input_video = 'road.mp4'
    output_video = 'output_video_detected'

    # 使用Ultralytics内置的预测接口，一行代码即可完成视频检测
    results = model.predict(source=input_video, save=True, conf=0.5, save_txt=False, project="", name=output_video)
    """
    model.predict 是 Ultralytics YOLO 模型的核心预测方法，其主要参数功能如下：
        - source: 指定输入数据源，可以是图像路径、视频路径、目录或URL等
        - save: 布尔值，控制是否将检测结果保存到文件（默认为 False）
        - conf: 置信度阈值，只保留置信度高于此值的检测结果（默认为 0.25）
        - save_txt: 布尔值，控制是否将检测结果以文本格式保存（默认为 False）
        - project: 指定保存结果的项目目录路径（默认为 "runs/detect"）
        - name: 指定保存结果的子目录名称（默认为 "exp"）
    """

