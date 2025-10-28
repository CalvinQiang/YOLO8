from ultralytics import YOLO
import cv2

if __name__ == '__main__':
    # 加载模型
    model = YOLO('../0-models/yolov8n.pt')  # 使用nano模型保证实时性

    # 打开默认摄像头（ID=0）
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 对摄像头画面进行实时检测
        results = model(frame, conf=0.7, verbose=False)  # verbose=False减少输出噪音
        # 绘制结果
        annotated_frame = results[0].plot()
        # 显示画面
        cv2.imshow('Real-time Camera Detection', annotated_frame)
        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()