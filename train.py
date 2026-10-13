from ultralytics import YOLO

# 加载YOLOv8轻量分类模型
model = YOLO('yolov8n-cls.pt')

# 启动训练
model.train(
    data='./cifar10',  # 数据集路径
    epochs=200,        # 训练轮数，和投影图对应
    imgsz=32,          # CIFAR-10原生图片尺寸
    batch=128,
    device='cpu',
    # 有NVIDIA显卡改成0，速度快很多
)