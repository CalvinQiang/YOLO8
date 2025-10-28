import cv2
from ultralytics import YOLO


def main():
    # 1. 加载YOLO模型（请确保模型文件路径正确）
    # 可选模型: 'yolov8n.pt'（最快）, 'yolov8s.pt', 'yolov8m.pt'（更准但更慢）等
    model = YOLO('../0-models/yolov8n.pt')  # 使用Nano版本平衡速度和精度

    # 2. 打开输入视频文件
    input_video_path = 'oceans.mp4'  # 请替换为你的视频路径
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print("错误：无法打开视频文件。请检查路径是否正确。")
        return

    # 3. 获取原视频的属性，用于设置输出视频
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 4. 创建VideoWriter对象以保存结果视频
    # 定义编解码器并创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或者尝试 'avc1'
    output_video_path = 'output_oceans.mp4'
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f"开始处理视频，总帧数: {total_frames}, 分辨率: {width}x{height}, FPS: {fps}")
    frame_count = 0

    # 5. 循环读取并处理每一帧
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # 如果读取失败（如到文件末尾），退出循环

        # 6. 使用YOLO模型对当前帧进行预测
        # 使用model.track可以实现物体追踪
        results = model(frame)

        # 7. 在帧上绘制检测结果（自动绘制边界框和标签）
        annotated_frame = results[0].plot()

        # 8. 将处理后的帧写入输出视频
        out.write(annotated_frame)

        # 9. （可选）在窗口中实时显示结果
        cv2.imshow('YOLOv8 Video Detection', annotated_frame)

        frame_count += 1
        if frame_count % 30 == 0:  # 每处理30帧打印一次进度
            print(f'处理进度: {frame_count}/{total_frames} ({frame_count / total_frames * 100:.1f}%)')

        # 10. 允许用户按'q'键提前退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 11. 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"处理完成！结果已保存至: {output_video_path}")

if __name__ == "__main__":
    main()